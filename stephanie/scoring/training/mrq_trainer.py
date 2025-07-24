from datetime import datetime

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.model import MRQModel
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.models.training_stats import TrainingStatsORM


class MRQTrainer(BaseTrainer):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.use_tuner = cfg.get("use_tuner", True)
        self.min_samples = cfg.get("min_samples", 100)

        self.logger.log("MRQTrainerInitialized", {
            "embedding_type": self.embedding_type,
            "use_tuner": self.use_tuner,
            "device": str(self.device)
        })

    def _create_dataloader(self, samples):
        valid_samples = []
        for s in samples:
            context_text = s.get("title", "")
            doc_text = s.get("output", "")
            if not context_text or not doc_text:
                continue
            try:
                context_emb = torch.tensor(self.memory.embedding.get_or_create(context_text)).to(self.device)
                doc_emb = torch.tensor(self.memory.embedding.get_or_create(doc_text)).to(self.device)
                score = float(s.get("score", 0.5))
                valid_samples.append({
                    "context": context_emb,
                    "document": doc_emb,
                    "score": score
                })
            except Exception:
                continue

        if len(valid_samples) < self.min_samples:
            self.logger.log("InsufficientSamples", {
                "dimension": self.dimension,
                "sample_count": len(valid_samples),
                "threshold": self.min_samples
            })
            return None

        context_tensors = torch.stack([s["context"] for s in valid_samples]).to(self.device)
        doc_tensors = torch.stack([s["document"] for s in valid_samples]).to(self.device)
        scores = torch.tensor([s["score"] for s in valid_samples]).to(self.device)

        dataset = TensorDataset(context_tensors, doc_tensors, scores)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _build_model(self):
        encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
        predictor = ValuePredictor(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        return MRQModel(encoder, predictor, self.memory.embedding, device=self.device)

    def _train_epoch(self, model, dataloader):
        model.train()
        total_loss, count = 0.0, 0
        for ctx_emb, doc_emb, scores in dataloader:
            ctx_emb, doc_emb, scores = ctx_emb.to(self.device), doc_emb.to(self.device), scores.to(self.device)
            zsa = model.encoder(ctx_emb, doc_emb)
            predictions = model.predictor(zsa).squeeze()
            loss = F.mse_loss(predictions, scores)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item() * ctx_emb.size(0)
            count += ctx_emb.size(0)
        return total_loss / count

    def train(self, samples, dimension):
        self.set_active_dimension(dimension)
        dataloader = self._create_dataloader(samples)
        if not dataloader:
            return {"error": "insufficient_data", "dimension": dimension}

        model = self._build_model()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        best_loss = float("inf")
        early_stop_counter = 0
        losses = []

        for epoch in range(self.epochs):
            avg_loss = self._train_epoch(model, dataloader)
            losses.append(avg_loss)
            self.logger.log("MRQTrainingEpoch", {
                "epoch": epoch + 1,
                "loss": avg_loss
            })
            if avg_loss < best_loss - self.early_stopping_min_delta:
                best_loss = avg_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if self.use_early_stopping and early_stop_counter >= self.early_stopping_patience:
                    break

        torch.save(model.encoder.state_dict(), self.locator.encoder_file())
        torch.save(model.predictor.state_dict(), self.locator.predictor_file())

        if self.use_tuner:
            tuner = RegressionTuner(dimension=dimension, logger=self.logger)
            for ctx_emb, doc_emb, scores in dataloader:
                zsa = model.encoder(ctx_emb.to(self.device), doc_emb.to(self.device))
                preds = model.predictor(zsa).squeeze().detach().cpu().numpy()
                actuals = scores.cpu().numpy()
                for p, a in zip(preds, actuals):
                    tuner.train_single(float(p), float(a))
            tuner.save(self.locator.tuner_file())

        scores_np = torch.tensor([s["score"] for s in samples])
        min_score = float(torch.min(scores_np))
        max_score = float(torch.max(scores_np))

        meta = {
            "dimension": dimension,
            "model_type": "mrq",
            "target_type": self.target_type,
            "embedding_type": self.embedding_type,
            "version": self.version,
            "dim": self.dim,
            "hdim": self.hdim,
            "min_score": min_score,
            "max_score": max_score,
            "avg_loss": best_loss,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._save_meta_file(meta, dimension)

        training_stat = TrainingStatsORM(
            model_type="mrq",
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            avg_q_loss=best_loss
        )
        self.memory.session.add(training_stat)
        self.memory.session.commit()

        return meta
