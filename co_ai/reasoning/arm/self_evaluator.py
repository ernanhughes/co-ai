import torch
import torch.nn.functional as F

from co_ai.reasoning.arm.utils import detect_format
from co_ai.evaluator import TextEncoder, HypothesisValuePredictor, MRQSelfEvaluator


class ARMReasoningSelfEvaluator(MRQSelfEvaluator):
    def __init__(self, memory, logger, device="cpu"):
        super().__init__(memory, logger, device)

        # Format usage tracking
        self.format_freq = {}
        self.format_rewards = {}

        # Training hyperparameters
        self.kl_penalty_coeff = 0.05
        self.beta = 0.1
        self.epsilon = 0.2
        self.decay_steps = 1000
        self.current_step = 0

        # ARM-specific modules
        self.encoder = TextEncoder().to(device)
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)

    def _score_response(self, prompt_emb, response_emb):
        """Score a single response using prompt-response encoder + value predictor"""
        zsa = self.encoder(prompt_emb, response_emb)
        return self.value_predictor(zsa), zsa

    def judge(self, goal, prompt, output_a, output_b):
        """
        Returns the preferred output + score details.
        Now includes format awareness and rarity-based scaling.
        """
        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt), device=self.device
        ).unsqueeze(0)
        output_a_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_a), device=self.device
        ).unsqueeze(0)
        output_b_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_b), device=self.device
        ).unsqueeze(0)

        # Score both responses
        value_a, _ = self._score_response(prompt_emb, output_a_emb)
        value_b, _ = self._score_response(prompt_emb, output_b_emb)

        # Convert to float
        value_a = value_a.item()
        value_b = value_b.item()

        # Detect formats
        fmt_a = detect_format(output_a)
        fmt_b = detect_format(output_b)

        # Penalize long CoT slightly
        token_len_a = len(output_a.split())
        token_len_b = len(output_b.split())

        value_a -= 0.01 * token_len_a
        value_b -= 0.01 * token_len_b

        # Apply format rarity bonus
        rarity_bonus_a = 1.0 / (1 + self.format_freq.get(fmt_a, 0))
        rarity_bonus_b = 1.0 / (1 + self.format_freq.get(fmt_b, 0))

        value_a += rarity_bonus_a
        value_b += rarity_bonus_b

        # Update internal counters
        self._update_format_stats(fmt_a, value_a)
        self._update_format_stats(fmt_b, value_b)

        better_output = output_a if value_a >= value_b else output_b
        better_fmt = fmt_a if value_a >= value_b else fmt_b

        return better_output, {
            "value_a": value_a,
            "value_b": value_b,
            "fmt_a": fmt_a,
            "fmt_b": fmt_b,
            "chosen_format": better_fmt,
            "token_len_a": token_len_a,
            "token_len_b": token_len_b
        }

    def _update_format_stats(self, fmt: str, reward: float):
        """Track format usage and average reward per format."""
        if fmt not in self.format_freq:
            self.format_freq[fmt] = 0
            self.format_rewards[fmt] = []

        self.format_freq[fmt] += 1
        self.format_rewards[fmt].append(reward)

    def _format_diversity_weight(self, responses: list[str], rewards: list[float]) -> list[float]:
        """
        Scale rewards based on format rarity and decay over time.
        Implements Ada-GRPO-style diversity scaling.
        """
        formats = [detect_format(r) for r in responses]
        freq = {}
        for f in formats:
            freq[f] = freq.get(f, 0) + 1

        weights = []
        for i in range(len(responses)):
            fmt = formats[i]
            G = len(responses)
            F = freq[fmt]

            diversity_factor = G / F
            decay_factor = (1 - self.current_step / self.decay_steps) * 0.8 + 0.2
            weight = diversity_factor * decay_factor * rewards[i]
            weights.append(weight)

        return weights

    def train_from_database(self, goal: str, cfg: dict):
        """
        Trains the value predictor using preference pairs stored in memory.
        Includes KL penalty and format-aware reward shaping.
        """
        limit = cfg.get("limit", 1000)
        epochs = cfg.get("epochs", 20)
        lr = cfg.get("lr", 1e-4)
        batch_size = cfg.get("batch_size", 16)

        samples = self.memory.mrq.get_training_pairs(goal=goal, limit=limit)
        if not samples or len(samples) == 0:
            print("[ERROR] No training samples found.")
            return

        inputs, labels = [], []
        for item in samples:
            prompt_emb = self.memory.embedding.get_or_create(item["prompt"])
            output_a_emb = self.memory.embedding.get_or_create(item["output_a"])
            output_b_emb = self.memory.embedding.get_or_create(item["output_b"])

            zsa_a = self.encoder(
                torch.tensor(prompt_emb).unsqueeze(0),
                torch.tensor(output_a_emb).unsqueeze(0)
            )
            zsa_b = self.encoder(
                torch.tensor(prompt_emb).unsqueeze(0),
                torch.tensor(output_b_emb).unsqueeze(0)
            )

            diff = zsa_a - zsa_b if item["preferred"] == "a" else zsa_b - zsa_a
            inputs.append(diff.squeeze(0).detach())
            labels.append(torch.tensor([1.0]))

        dataset = torch.utils.data.TensorDataset(torch.stack(inputs), torch.stack(labels))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt = torch.optim.Adam(self.value_predictor.parameters(), lr=lr)
        self.value_predictor.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataloader:
                preds = self.value_predictor(x_batch)
                policy_log_probs = torch.log_softmax(preds, dim=-1)

                with torch.no_grad():
                    ref_preds = self.ref_value_predictor(x_batch)
                    ref_log_probs = torch.log_softmax(ref_preds, dim=-1)

                # Compute advantages
                advantages = policy_log_probs - ref_log_probs
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

                # Clip ratios
                ratios = torch.exp(policy_log_probs - ref_log_probs)
                unclipped_loss = ratios * advantages
                clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                clipped_loss = clipped_ratios * advantages

                policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()
                kl = F.kl_div(ref_log_probs, policy_log_probs, reduction='batchmean')

                loss = policy_loss + self.kl_penalty_coeff * kl

                loss.backward()
                opt.step()
                opt.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.logger.log("TrainingEpoch", {
                "epoch": epoch + 1,
                "avg_loss": round(avg_loss, 5),
                "goal": goal,
                "format_usage": self.format_freq.copy(),
                "format_rewards": {k: sum(v)/len(v) for k, v in self.format_rewards.items()}
            })

            self.current_step += 1

        self.logger.log("TrainingComplete", {"goal": goal})

def select_best_format(self, prompt: str, options: dict[str, str]):
    prompt_emb = torch.tensor(
        self.memory.embedding.get_or_create(prompt), device=self.device
    ).unsqueeze(0)

    scores = {}
    for fmt, response in options.items():
        response_emb = torch.tensor(
            self.memory.embedding.get_or_create(response), device=self.device
        ).unsqueeze(0)

        value, _ = self._score_response(prompt_emb, response_emb)
        token_len = len(response.split())
        rarity_bonus = 1.0 / (1 + self.format_freq.get(fmt, 0))

        final_score = value.item() - 0.01 * token_len + rarity_bonus
        scores[fmt] = final_score

    best_format = max(scores, key=scores.get)
    return best_format, scores