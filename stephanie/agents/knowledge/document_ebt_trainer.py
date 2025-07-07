import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stephanie.agents.base_agent import BaseAgent
from stephanie.evaluator.text_encoder import TextEncoder
from stephanie.scoring.document_pair_builder import \
    DocumentPreferencePairBuilder
from stephanie.scoring.document_value_predictor import DocumentValuePredictor


class DocumentEBTDataset(Dataset):
    def __init__(self, contrast_pairs):
        self.data = []
        for pair in contrast_pairs:
            context = pair.get("title", "")
            self.data.append((context, pair["output_a"], -pair["value_a"]))  # Lower is better
            self.data.append((context, pair["output_b"], -pair["value_b"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DocumentEBTScorer(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, ctx_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([ctx_emb, doc_emb], dim=-1)
        return self.head(combined).squeeze(-1)


class DocumentEBTTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_save_path = cfg.get("model_save_path", "models")
        self.model_prefix = cfg.get("model_prefix", "document_ebt_")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")

        builder = DocumentPreferencePairBuilder(db=self.memory.session, logger=self.logger)
        training_pairs = builder.get_training_pairs_by_dimension(goal=goal_text)

        os.makedirs(self.model_save_path, exist_ok=True)

        for dim, pairs in training_pairs.items():
            if not pairs:
                continue

            self.logger.log("DocumentEBTTrainingStart", {"dimension": dim, "num_pairs": len(pairs)})

            ds = DocumentEBTDataset(pairs)
            dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=self.collate_ebt_batch)

            model = DocumentEBTScorer().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
            loss_fn = nn.MSELoss()

            for epoch in range(5):
                model.train()
                total_loss = 0.0
                for ctx_enc, cand_enc, labels in dl:
                    preds = model(ctx_enc, cand_enc)
                    loss = loss_fn(preds, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(dl)
                self.logger.log("DocumentEBTEpoch", {
                    "dimension": dim,
                    "epoch": epoch + 1,
                    "avg_loss": round(avg_loss, 5)
                })

            path = os.path.join(self.model_save_path, f"{self.model_prefix}{dim}.pt")
            torch.save(model.state_dict(), path)
            self.logger.log("DocumentEBTModelSaved", {"dimension": dim, "path": path})

        context[self.output_key] = training_pairs
        return context

    def collate_ebt_batch(self, batch):
        ctxs, docs, targets = zip(*batch)

        ctx_embs = [torch.tensor(self.memory.embedding.get_or_create(c)).to(self.device) for c in ctxs]
        doc_embs = [torch.tensor(self.memory.embedding.get_or_create(d)).to(self.device) for d in docs]
        labels = torch.tensor(targets, dtype=torch.float32).to(self.device)

        ctx_tensor = torch.stack(ctx_embs)
        doc_tensor = torch.stack(doc_embs)

        return ctx_tensor, doc_tensor, labels
