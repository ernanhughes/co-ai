import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from stephanie.agents.base_agent import BaseAgent
from stephanie.evaluator.text_encoder import TextEncoder
from stephanie.scoring.document_value_predictor import DocumentValuePredictor
from stephanie.utils.model_utils import get_model_path
from stephanie.utils.file_utils import save_json
from stephanie.scoring.model.ebt_model import EBTModel


class DocumentEBTDataset(Dataset):
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
            norm_a = (pair["value_a"] - self.min_score) / (self.max_score - self.min_score)
            norm_b = (pair["value_b"] - self.min_score) / (self.max_score - self.min_score)
            self.data.append((pair["title"], pair["output_a"], norm_a))
            self.data.append((pair["title"], pair["output_b"], norm_b))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def get_normalization(self):
        # Returns score range so inference can denormalize output later
        return {"min": self.min_score, "max": self.max_score}


class DocumentEBTTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = "ebt"
        self.target_type = "document"
        self.encoder = TextEncoder().to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.value_predictor = DocumentValuePredictor().to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")

        from stephanie.scoring.document_pair_builder import (
            DocumentPreferencePairBuilder,
        )

        # Build contrastive training pairs grouped by scoring dimension
        builder = DocumentPreferencePairBuilder(
            db=self.memory.session, logger=self.logger
        )
        training_pairs = builder.get_training_pairs_by_dimension(goal=goal_text)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train one model per scoring dimension (e.g. clarity, novelty, etc.)
        for dim, pairs in training_pairs.items():
            if not pairs:
                continue

            self.logger.log("DocumentEBTTrainingStart", {"dimension": dim, "num_pairs": len(pairs)})

            # Construct dataset and dataloader; normalize scores between 50–100
            ds = DocumentEBTDataset(pairs, min_score=50, max_score=100)
            dl = DataLoader(
                ds,
                batch_size=8,
                shuffle=True,
                collate_fn=lambda b: collate_ebt_batch(b, self.memory.embedding, device)
            )

            # Create model for this dimension
            model = EBTModel().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
            loss_fn = nn.MSELoss()

            # Training loop for fixed number of epochs
            for epoch in range(10):
                model.train()
                total_loss = 0.0
                for ctx_enc, cand_enc, labels in dl:
                    preds = model(ctx_enc, cand_enc)  # Predict score given (goal, doc)
                    loss = loss_fn(preds, labels)      # Compare against normalized label

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(dl)
                self.logger.log("DocumentEBTEpoch", {"dimension": dim, "epoch": epoch + 1, "avg_loss": round(avg_loss, 5)})

            # Save trained model weights to disk
            model_path = f"{get_model_path(self.model_type, self.target_type, dim)}.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(model.state_dict().keys())
            torch.save(model.state_dict(), model_path)
            self.logger.log("DocumentEBTModelSaved", {"dimension": dim, "path": model_path})

            # Save score normalization metadata for this dimension
            meta_path = model_path.replace(".pt", ".meta.json")
            normalization = ds.get_normalization()
            save_json(normalization, meta_path)

        context[self.output_key] = training_pairs
        return context


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
