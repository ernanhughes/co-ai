# stephanie/agents/maintenance/ebt_trainer.py
import os
import sys
import json

import torch
from sqlalchemy import text
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.maintenance.model_evolution_manager import \
    ModelEvolutionManager
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.utils.model_utils import save_model_with_version
from stephanie.utils.model_locator import ModelLocator
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np
from stephanie.scoring.transforms.regression_tuner import RegressionTuner

class EBTDataset(Dataset):
    def __init__(self, contrast_pairs, min_score=None, max_score=None):
        self.data = []

        # Compute min/max from all pair values if not explicitly provided
        all_scores = []
        for pair in contrast_pairs:
            all_scores.extend([pair["value_a"], pair["value_b"]])
        self.min_score = min(all_scores) if min_score is None else min_score
        self.max_score = max(all_scores) if max_score is None else max_score

        # Normalize scores and store training examples as (goal, document, normalized_score)
        for pair in contrast_pairs:
            norm_a = (pair["value_a"] - self.min_score) / (
                self.max_score - self.min_score
            )
            norm_b = (pair["value_b"] - self.min_score) / (
                self.max_score - self.min_score
            )
            self.data.append((pair["title"], pair["output_a"], norm_a))
            self.data.append((pair["title"], pair["output_b"], norm_b))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def get_normalization(self):
        # Returns score range so inference can denormalize output later
        return {"min": self.min_score, "max": self.max_score}


class EBTTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_path = cfg.get("model_path", "models")
        self.model_type = cfg.get("model_type", "ebt")
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")

        self.early_stopping_patience = cfg.get("early_stopping_patience", 3)
        self.early_stopping_min_delta = cfg.get("early_stopping_min_delta", 1e-4)
        self.early_stop_counter = 0
        self.best_loss = float('inf')
        self.early_stop_active = False
        self.early_stop_threshold = None

        self.embedding_type = self.memory.embedding.type
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim
        self.num_actions = cfg.get("num_actions", 3)
        self.dimensions = cfg.get("dimensions", [])
        self.device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")


        self.encoder = TextEncoder().to(self.device)
        self.value_predictor = ValuePredictor().to(self.device)
        self.evolution_manager = ModelEvolutionManager(
            self.cfg, self.memory, self.logger
        )
        self.tuners = {}
        self._load_tuners()
    
    def _load_tuners(self):
        """Load tuners for each dimension"""
        for dim in self.dimensions:
            locator = ModelLocator(
                root_dir=self.model_path,
                dimension=dim,
                model_type=self.model_type,
                target_type=self.target_type,
                version=self.model_version,
                embedding_type=self.embedding_type
            )
            tuner_path = locator.tuner_file()
            if os.path.exists(tuner_path):
                self.tuners[dim] = RegressionTuner(dim=dim)
                self.tuners[dim].load(tuner_path)
            else:
                self.tuners[dim] = None
                self.logger.log("TunerMissing", {
                    "dimension": dim,
                    "path": tuner_path
                })

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")

        from stephanie.scoring.mrq.preference_pair_builder import \
            PreferencePairBuilder

        # Build contrastive training pairs grouped by scoring dimension
        builder = PreferencePairBuilder(db=self.memory.session, logger=self.logger)

        # Train one model per scoring dimension (e.g. clarity, novelty, etc.)
        for dim in self.dimensions:
            training_pairs = builder.get_training_pairs_by_dimension(goal=goal_text, dim=[dim])
            self.logger.log(
                "DocumentEBTTrainingStart", {"dimension": dim, "num_pairs": len(training_pairs)}
            )
            for dim, pairs in training_pairs.items():
                if not pairs:
                    self.logger.log(
                        "DocumentEBTTrainingSkipped", {"dimension": dim, "reason": "No pairs"}
                    )
                    continue

                self.logger.log(
                    "DocumentEBTTrainingStart", {"dimension": dim, "num_pairs": len(pairs)}
                )

                # Construct dataset and dataloader; normalize scores between 50–100
                ds = EBTDataset(pairs, min_score=1, max_score=100)
                dl = DataLoader(
                    ds,
                    batch_size=8,
                    shuffle=True,
                    collate_fn=lambda b: collate_ebt_batch(
                        b, self.memory.embedding, self.device
                    ),
                )

                # Create model for this dimension
                model = EBTModel(self.dim, self.hdim, self.num_actions, self.device).to(self.device)
                self.train_ebt_model(model, dl, dim)    
                self._save_model(model, dim)

        context[self.output_key] = training_pairs
        return context


    # Updated training loop
    def train_ebt_model(self, model: EBTModel, dl: DataLoader, dim: str):
        """Train EBTModel with Q/V/Policy heads"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.get("lr", 2e-5))
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        
        # Loss functions
        mse = nn.MSELoss()
        def expectile_loss(diff, tau=0.7):
            return (torch.where(diff > 0, tau * diff.pow(2), (1 - tau) * diff.pow(2))).mean()
        
        # Training stats
        stats = {
            "q_losses": [],
            "v_losses": [],
            "pi_losses": [],
            "total_losses": [],
            "policy_entropies": []
        }

        for epoch in range(self.cfg.get("epochs", 10)):
            epoch_q_loss = 0.0
            epoch_v_loss = 0.0
            epoch_pi_loss = 0.0
            
            for ctx_enc, cand_enc, labels in dl:
                # Ensure device alignment
                ctx_enc = ctx_enc.to(self.device)
                cand_enc = cand_enc.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(ctx_enc, cand_enc)
                
                # Q-head loss (supervised learning)
                q_loss = mse(outputs["q_value"], labels)
                
                # V-head loss (expectile regression)
                v_loss = expectile_loss(outputs["q_value"].detach() - outputs["state_value"])
                
                # Policy head loss (AWR)
                advantage = (outputs["q_value"] - outputs["state_value"]).detach()
                policy_probs = F.softmax(outputs["action_logits"], dim=-1)
                entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=-1).mean()
                stats["policy_entropies"].append(entropy.item())
                pi_loss = -(torch.log(policy_probs) * advantage).mean() - 0.01 * entropy  # With entropy regularization
                
                # Composite loss
                total_loss = (
                    q_loss * self.cfg.get("q_weight", 1.0) +
                    v_loss * self.cfg.get("v_weight", 0.5) +
                    pi_loss * self.cfg.get("pi_weight", 0.3)
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Track losses
                epoch_q_loss += q_loss.item()
                epoch_v_loss += v_loss.item()
                epoch_pi_loss += pi_loss.item()
            
            # End of epoch
            avg_q_loss = epoch_q_loss / len(dl)
            avg_v_loss = epoch_v_loss / len(dl)
            avg_pi_loss = epoch_pi_loss / len(dl)
            
            stats["q_losses"].append(avg_q_loss)
            stats["v_losses"].append(avg_v_loss)
            stats["pi_losses"].append(avg_pi_loss)
            stats["total_losses"].append(avg_q_loss + avg_v_loss + avg_pi_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_q_loss)
            
            entropy_value = (
                torch.mean(torch.tensor(stats["policy_entropies"])).item()
                if stats["policy_entropies"]
                else float("nan")  # or 0.0 if you prefer
            )

            self.logger.log("EBTModelTrainerEpoch", {
                "dimension": dim,
                "epoch": epoch + 1,
                "q_loss": avg_q_loss,
                "v_loss": avg_v_loss,
                "pi_loss": avg_pi_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "policy_entropy": entropy_value,
            })

            # Early stopping
            if self._should_stop_early(stats["q_losses"]):
                self.logger.log("EBTModelTrainerEarlyStopping", {
                    "dimension": dim,
                    "epoch": epoch + 1
                })
                break

        return {
            "q_loss": stats["q_losses"][-1],
            "v_loss": stats["v_losses"][-1],
            "pi_loss": stats["pi_losses"][-1],
            "avg_q_loss": np.mean(stats["q_losses"]),
            "avg_v_loss": np.mean(stats["v_losses"]),
            "avg_pi_loss": np.mean(stats["pi_losses"]),
            "policy_entropy": np.mean(stats["policy_entropies"]),
            "model": model
        }

    def _calculate_policy_entropy(self, model, data):
        """Calculate policy entropy with validation"""
        if not data:
            return 0.0  # Fallback if no data
        
        with torch.no_grad():
            context_embs = torch.stack([d["context_emb"] for d in data])
            doc_embs = torch.stack([d["doc_emb"] for d in data])
            
            outputs = model(context_embs.to(self.device), doc_embs.to(self.device))
            action_probs = F.softmax(outputs["action_logits"], dim=-1)
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        
        return entropy.mean().item() if len(entropy) > 0 else 0.0

    def _save_model(self, model, dim):
        locator = ModelLocator(
            root_dir=self.model_path,
            embedding_type=self.embedding_type,
            model_type="ebt",
            target_type="document",
            dimension=dim,
            version=self.model_version
        )
        
        # Save components
        locator.ensure_dirs()
        
        print(f"Saving EBT model to {locator.encoder_file()}")
        torch.save(model.encoder.state_dict(), locator.encoder_file())
        torch.save(model.q_head.state_dict(), locator.get_q_head_path())
        torch.save(model.v_head.state_dict(), locator.get_v_head_path())
        torch.save(model.pi_head.state_dict(), locator.get_pi_head_path())
        
        # Save meta
        meta = {
            "dim": model.embedding_dim,
            "hdim": model.hidden_dim,
            "num_actions": model.num_actions,
            "version": self.model_version,
            "scorer_map": ["ebt", "svm", "mrq"]
        }
        with open(locator.meta_file(), "w") as f:
            json.dump(meta, f)
        
        # Save tuner
        if self.tuners[dim]:
            self.tuners[dim].save(locator.tuner_file())

    def _save_and_promote_model(self, model, model_type, target_type, dimension):
        # Generate new version ID
        version = self._generate_version(model_type, target_type, dimension)

        # Save model with version
        version_path = save_model_with_version(
            model.state_dict(), model_type, target_type, dimension, version
        )

        # Log in DB
        model_id = self.evolution_manager.log_model_version(
            model_type=model_type,
            target_type=target_type,
            dimension=dimension,
            version=version,
            performance=self._get_validation_metrics(),  # e.g., accuracy, loss
        )

        # Get current best model
        current = self.evolution_manager.get_best_model(
            model_type, target_type, dimension
        )

        # Compare performance and promote if better
        if self.evolution_manager.check_model_performance(
            new_perf=self._get_validation_metrics(),
            old_perf=current["performance"] if current else {},
        ):
            self.evolution_manager.promote_model_version(model_id)
            self.logger.log(
                "ModelPromoted",
                {
                    "model_type": model_type,
                    "dimension": dimension,
                    "version": version,
                    "path": version_path,
                },
            )
        else:
            self.logger.log(
                "ModelNotPromoted",
                {
                    "model_type": model_type,
                    "dimension": dimension,
                    "new_version": version,
                    "current_version": current["version"] if current else None,
                },
            )

    def _generate_version(self, model_type, target_type, dimension):
        return "v1"

    def _get_validation_metrics(self) -> dict:
        """
        Compute validation metrics (loss, accuracy, etc.) from scoring history.
        This serves as the model's performance snapshot.
        """
        query = """
        SELECT raw_score, transformed_score
        FROM scoring_history
        WHERE model_type = :model_type
        AND target_type = :target_type
        """

        rows = self.memory.session.execute(
            text(query),
            {
                "model_type": self.model_type,
                "target_type": self.target_type,
            },
        ).fetchall()

        raw_scores = [row.raw_score for row in rows if row.raw_score is not None]
        transformed_scores = [
            row.transformed_score for row in rows if row.transformed_score is not None
        ]

        if len(raw_scores) < 2 or len(transformed_scores) < 2:
            return {"validation_loss": sys.float_info.max, "accuracy": 0.0}

        # Use mean squared error between raw and transformed scores as a proxy for loss
        squared_errors = [(r - t) ** 2 for r, t in zip(raw_scores, transformed_scores)]
        validation_loss = sum(squared_errors) / len(squared_errors)

        # Simple accuracy proxy: proportion of scores that are within a 0.1 margin
        correct_margin = sum(
            1 for r, t in zip(raw_scores, transformed_scores) if abs(r - t) <= 0.1
        )
        accuracy = correct_margin / len(raw_scores)

        return {
            "validation_loss": round(validation_loss, 4),
            "accuracy": round(accuracy, 4),
        }

    def _should_stop_early(self, losses):
        if not losses or len(losses) < self.early_stopping_patience:
            return False
        
        # Use sliding window of last N losses
        window = losses[-self.early_stopping_patience:]
        avg_window = sum(window) / len(window)
        
        # If no improvement in window
        if avg_window >= self.best_loss - self.early_stopping_min_delta:
            self.early_stop_counter += 1
            return self.early_stop_counter >= self.early_stopping_patience
        else:
            self.best_loss = avg_window
            self.early_stop_counter = 0
            return False

def collate_ebt_batch(batch, embedding_store, device):
    # Custom batch collation for EBT dataset: fetch embeddings for goal and doc
    ctxs, docs, targets = zip(*batch)

    # Look up or create embeddings for each goal and candidate doc
    ctx_embs = [torch.tensor(embedding_store.get_or_create(c)).to(device) for c in ctxs]
    doc_embs = [torch.tensor(embedding_store.get_or_create(d)).to(device) for d in docs]
    labels = torch.tensor(targets, dtype=torch.float32).to(device)

    # Stack them into batched tensors for training
    ctx_tensor = torch.stack(ctx_embs)
    doc_tensor = torch.stack(doc_embs)

    return ctx_tensor, doc_tensor, labels

