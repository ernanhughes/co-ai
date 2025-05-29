import torch
import torch.nn.functional as F
import torch.optim as optim
import json

from co_ai.evaluator import (HypothesisValuePredictor, MRQSelfEvaluator,
                             TextEncoder)
from co_ai.reasoning.arm.utils import detect_format


class ARMReasoningSelfEvaluator(MRQSelfEvaluator):
    def __init__(self, memory, logger, device="cpu"):
        super().__init__(memory, logger, device)

        # Format usage tracking
        self.format_freq = {"direct": 1, "short_cot": 1, "code": 1, "long_cot": 1}
        self.format_rewards = {
            "direct": [0.5],
            "short_cot": [0.5],
            "code": [0.5],
            "long_cot": [0.5],
        }

        # Training hyperparameters
        self.kl_penalty_coeff = 0.05
        self.beta = 0.1
        self.epsilon = 0.2
        self.decay_steps = 1000
        self.current_step = 0

        # ARM-specific modules
        self.encoder = TextEncoder().to(device)
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)
        self.ref_value_predictor = self._create_ref_model()

    def _create_ref_model(self):
        """Create a frozen copy of the value predictor as reference"""
        ref_model = HypothesisValuePredictor(512, 1024).to(self.device)
        ref_model.load_state_dict(self.value_predictor.state_dict())
        ref_model.requires_grad_(False)
        ref_model.eval()
        return ref_model

    def score(self, prompt: str, response: str) -> float:
        """
        Public scoring method used by agents like AdaptiveReasonerAgent.
        Returns a scalar score indicating how good a response is.
        """
        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt), device=self.device
        ).unsqueeze(0)
        response_emb = torch.tensor(
            self.memory.embedding.get_or_create(response), device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            zsa = self.encoder(prompt_emb, response_emb)
            score = self.value_predictor(zsa).item()

        token_len = len(response.split())
        rarity_bonus = 1.0 / (1 + self.format_freq.get(detect_format(response), 1))
        score -= 0.01 * token_len
        score += rarity_bonus

        return score

    def _score_response(self, prompt_emb, response_emb):
        """Score a single response using prompt-response encoder + value predictor"""
        zsa = self.encoder(prompt_emb, response_emb)
        return self.value_predictor(zsa), zsa

    def judge(self, goal, prompt, output_a, output_b):
        """
        Returns the preferred output + score details.
        Now includes format awareness and rarity-based scaling.
        """
        prompt_emb = torch.tensor(self.memory.embedding.get_or_create(prompt)).unsqueeze(0).to(self.device)
        output_a_emb = torch.tensor(self.memory.embedding.get_or_create(output_a)).unsqueeze(0).to(self.device)
        output_b_emb = torch.tensor(self.memory.embedding.get_or_create(output_b)).unsqueeze(0).to(self.device)

        # Score both responses
        value_a, _ = self._score_response(prompt_emb, output_a_emb)
        value_b, _ = self._score_response(prompt_emb, output_b_emb)

        # Convert to float
        value_a = value_a.item()
        value_b = value_b.item()

        # Detect formats
        fmt_a = detect_format(output_a)
        fmt_b = detect_format(output_b)

        if fmt_a == "unknown" or fmt_b == "unknown":
            print(f"[WARNING] Unknown format detected:\nA: {output_a[:100]}...\nB: {output_b[:100]}...")

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

        if fmt == "unknown":
            print(f"[WARNING] Unknown format detected in response:\n{fmt[:100]}...")

        self.format_freq[fmt] += 1
        self.format_rewards[fmt].append(reward)

    def _format_diversity_weight(self, responses: list[str], rewards: list[float]) -> list[float]:
        """
        Scale rewards based on format rarity + decay over time.
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

        # Clamp weights to prevent extreme values
        return torch.clamp(torch.tensor(weights), min=-10, max=10).tolist()

    def train_from_database(self, goal_text: str, cfg: dict):
        """
        Trains the value predictor using preference pairs stored in memory.
        Includes KL penalty and format-aware reward shaping.
        """
        limit = cfg.get("limit", 1000)
        epochs = cfg.get("epochs", 20)
        lr = cfg.get("lr", 1e-4)
        batch_size = cfg.get("batch_size", 16)

        samples = self.memory.mrq.get_training_preferece_pairs(
            goal=goal_text, limit=limit
        )
        if not samples or len(samples) == 0:
            print("[ERROR] No training samples found.")
            return

        self.export_samples_to_json(samples, "arm_preference_pairs.json")

        inputs, labels = [], []
        for item in samples:
            fmt_a = item["fmt_a"]
            fmt_b = item["fmt_b"]

            if fmt_a == "unknown" or fmt_b == "unknown":
                print(f"[WARNING] Malformed format detected:\nA: {item['output_a'][:100]}...\nB: {item['output_b'][:100]}...")
                continue


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
                print(
                    "x_batch:",
                    torch.isnan(x_batch).any().item(),
                    torch.isinf(x_batch).any().item(),
                )
                print(
                    "preds (before clamp):",
                    torch.isnan(self.value_predictor(x_batch)).any().item(),
                )
                preds = self.value_predictor(x_batch).clamp(min=-10, max=10)
                if torch.isnan(preds).any():
                    print("❌ NaN in preds - skipping batch")
                    continue

                policy_log_probs = torch.log_softmax(preds, dim=-1).clamp(min=-100)

                print("preds (after clamp):", torch.isnan(preds).any().item())
                print("log_softmax:", torch.isnan(policy_log_probs).any().item())


                with torch.no_grad():
                    ref_preds = self.ref_value_predictor(x_batch).clamp(min=-10, max=10)
                    ref_log_probs = torch.log_softmax(ref_preds, dim=-1).clamp(min=-100)

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
                torch.nn.utils.clip_grad_norm_(self.value_predictor.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.logger.log("TrainingEpoch", {
                "epoch": epoch + 1,
                "avg_loss": round(avg_loss, 5),
                "goal": goal_text,
                "format_usage": self.format_freq.copy(),
                "format_rewards": {k: sum(v)/len(v) if v else 0 for k, v in self.format_rewards.items()},
                "kl_penalty": avg_loss
            })
            self.current_step += 1

        self.logger.log("TrainingComplete", {"goal": goal_text})

    def select_best_format(self, prompt: str, options: dict[str, str]):
        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt), device=self.device
        ).unsqueeze(0)


        scores = {}
        for fmt, response in options.items():
            response_emb = torch.tensor(
                self.memory.embedding.get_or_create(response), device=self.device
            ).unsqueeze(0)

            base_score, _ = self._score_response(prompt_emb, response_emb)
            token_len = len(response.split())
            rarity_bonus = 1.0 / (1 + self.format_freq.get(fmt, 0))
            final_score = base_score.item() - 0.01 * token_len + rarity_bonus
            scores[fmt] = final_score

        best_format = max(scores, key=scores.get)
        return {
            "response": options[best_format],
            "scores": scores,
            "chosen_format": best_format
        }

    def train_from_context(self, context: dict, cfg: dict):
        """
        Trains the value predictor using DPO samples stored in the context.
        Applies format-aware reward shaping and KL penalty.
        """
        dpo_samples = context.get("dpo_samples", [])
        if not dpo_samples:
            self.logger.log(
                "TrainingError", {"message": "No DPO samples found in context."}
            )
            return

        self.logger.log(
            "TrainingStarted", {"sample_count": len(dpo_samples), "config": cfg}
        )

        inputs, labels = [], []

        # Extract preference data
        for item in dpo_samples:
            prompt_emb = self.memory.embedding.get_or_create(item["prompt"])
            output_a_emb = self.memory.embedding.get_or_create(item["chosen"])
            output_b_emb = self.memory.embedding.get_or_create(item["rejected"])

            zsa_a = self.encoder(
                torch.tensor(prompt_emb).unsqueeze(0).to(self.device),
                torch.tensor(output_a_emb).unsqueeze(0).to(self.device),
            )
            zsa_b = self.encoder(
                torch.tensor(prompt_emb).unsqueeze(0).to(self.device),
                torch.tensor(output_b_emb).unsqueeze(0).to(self.device),
            )

            diff = zsa_a - zsa_b if item["preferred_format"] == "a" else zsa_b - zsa_a
            inputs.append(diff.squeeze(0).detach())
            labels.append(torch.tensor([1.0]))

        dataset = torch.utils.data.TensorDataset(
            torch.stack(inputs), torch.stack(labels)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.get("batch_size", 16), shuffle=True
        )

        opt = optim.Adam(self.value_predictor.parameters(), lr=cfg.get("lr", 1e-4))
        self.value_predictor.train()

        epochs = cfg.get("epochs", 20)
        best_loss = float("inf")
        patience_counter = 0
        patience = cfg.get("patience", 3)

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataloader:
                preds = self.value_predictor(x_batch)
                policy_log_probs = torch.log_softmax(preds, dim=-1)

                with torch.no_grad():
                    ref_preds = self.ref_value_predictor(x_batch)
                    ref_log_probs = torch.log_softmax(ref_preds, dim=-1)

                advantages = policy_log_probs - ref_log_probs
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-6
                )

                ratios = torch.exp(policy_log_probs - ref_log_probs)
                clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                unclipped_loss = ratios * advantages
                clipped_loss = clipped_ratios * advantages

                policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()
                kl = F.kl_div(ref_log_probs, policy_log_probs, reduction="batchmean")
                loss = policy_loss + self.kl_penalty_coeff * kl

                loss.backward()
                opt.step()
                opt.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.logger.log(
                "TrainingEpoch",
                {
                    "epoch": epoch + 1,
                    "avg_loss": round(avg_loss, 5),
                    "goal": "arm_dpo",
                    "format_usage": self.format_freq.copy(),
                    "format_rewards": {
                        k: round(sum(v) / len(v), 5) if v else 0
                        for k, v in self.format_rewards.items()
                    },
                },
            )

            if avg_loss < best_loss - 0.0001:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.log(
                    "EarlyStopping",
                    {"stopped_epoch": epoch + 1, "best_loss": round(best_loss, 5)},
                )
                break

        self.logger.log(
            "TrainingComplete",
            {"total_epochs": epoch + 1, "final_loss": round(avg_loss, 5)},
        )




    def export_samples_to_json(self, samples: list, output_path: str):
        """
        Exports raw preference pairs to a structured JSON file.

        Each entry includes:
            - Prompt
            - Output A / B
            - Format A / B
            - Preferred side
            - Token lengths
            - Rarity bonuses
            - Difficulty level
        """
        processed = []

        for item in samples:
            prompt = item.get("prompt", "")
            output_a = item.get("output_a", "")
            output_b = item.get("output_b", "")
            preferred = item.get("preferred", "a")

            # Detect format types
            fmt_a = detect_format(output_a)
            fmt_b = detect_format(output_b)

            # Count tokens
            token_len_a = len(output_a.split())
            token_len_b = len(output_b.split())

            # Add rarity bonus
            G = len(samples)
            F_a = (
                sum(1 for s in samples if detect_format(s.get("output_a", "")) == fmt_a) + 1
            )
            F_b = (
                sum(1 for s in samples if detect_format(s.get("output_b", "")) == fmt_b) + 1
            )

            rarity_bonus_a = G / F_a
            rarity_bonus_b = G / F_b

            # Infer difficulty from question length
            words = prompt.split()
            if len(words) < 20:
                difficulty = "easy"
            elif len(words) < 50:
                difficulty = "medium"
            else:
                difficulty = "hard"

            processed.append(
                {
                    "prompt": prompt,
                    "output_a": output_a,
                    "output_b": output_b,
                    "preferred": preferred,
                    "fmt_a": fmt_a,
                    "fmt_b": fmt_b,
                    "token_len_a": token_len_a,
                    "token_len_b": token_len_b,
                    "rarity_bonus_a": round(rarity_bonus_a, 3),
                    "rarity_bonus_b": round(rarity_bonus_b, 3),
                    "difficulty": difficulty,
                }
            )

        with open(output_path, "w") as fp:
            json.dump(processed, fp, indent=2)

        print(f"[INFO] Exported {len(processed)} samples to {output_path}")