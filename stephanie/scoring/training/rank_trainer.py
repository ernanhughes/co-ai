import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.scoring.model.preference_ranker import PreferenceRanker

class ContrastiveRankerTrainer(BaseTrainer):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.embedding_type = self.memory.embedding.type
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim
        self.root_dir = cfg.get("model_path", "models")
        self.dimension = cfg.get("dimension", "alignment")
        self.embedding_type = cfg.get("embedding_type", "hnet")
        self.model_type = "contrastive_ranker"
        self.target_type = cfg.get("target_type", "document")
        self.version = cfg.get("model_version", "v1")

        self.hidden_dim = cfg.get("hidden_dim", 256)
        self.lr = cfg.get("lr", 1e-3)
        self.epochs = cfg.get("epochs", 20)
        self.batch_size = cfg.get("batch_size", 64)
        self.patience = cfg.get("patience", 5)
        self.baseline = cfg.get("baseline", "I don't know how to answer this.")
        self.min_pairs = cfg.get("min_pairs", 10)  # Minimum pairs for training

    def train(self, samples, dimension):
        """Train pairwise preference model with baseline calibration"""
        # Validate sufficient data
        if len(samples) < self.min_pairs:
            return {"error": "insufficient_data", "dimension": dimension, "required": self.min_pairs, "found": len(samples)}

        # Process samples into pairwise format
        X_a, X_b, y = [], [], []
        absolute_scores = []  # For regression tuning
        
        for pair in samples:
            # Get embeddings (goal + output)
            ctx_emb = self.memory.embedding.get_or_create(pair["title"])
            a_emb = self.memory.embedding.get_or_create(pair["output_a"])
            b_emb = self.memory.embedding.get_or_create(pair["output_b"])
            
            # Combine context and output embeddings
            input_a = np.concatenate([ctx_emb, a_emb])
            input_b = np.concatenate([ctx_emb, b_emb])
            
            X_a.append(input_a)
            X_b.append(input_b)
            
            # Label: 1 if A preferred, 0 if B preferred
            y.append(1 if pair["preferred"] == "A" else 0)
            
            # Collect absolute scores for calibration
            if "value_a" in pair:
                absolute_scores.append((pair["output_a"], pair["title"], pair["value_a"]))
            if "value_b" in pair:
                absolute_scores.append((pair["output_b"], pair["title"], pair["value_b"]))

        # Convert to numpy arrays
        X_a = np.array(X_a)
        X_b = np.array(X_b)
        y = np.array(y)
        
        # Standardize inputs (critical for neural nets)
        all_X = np.vstack([X_a, X_b])
        scaler = StandardScaler()
        scaler.fit(all_X)
        
        X_a_scaled = scaler.transform(X_a)
        X_b_scaled = scaler.transform(X_b)

        # Initialize model
        input_dim = X_a_scaled.shape[1]
        model = PreferenceRanker(embedding_dim=input_dim, hidden_dim=self.hidden_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Convert to PyTorch tensors
        X_a_tensor = torch.tensor(X_a_scaled, dtype=torch.float32)
        X_b_tensor = torch.tensor(X_b_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_a_tensor, X_b_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        # Train with early stopping
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            
            for xa, xb, labels in dataloader:
                optimizer.zero_grad()
                logits = model(xa, xb)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Save model and scaler
        locator = self.get_locator(dimension)
        model_path = locator.model_file(suffix=".pt")
        scaler_path = locator.scaler_file(suffix=".joblib")
        
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)
        
        # Calibrate using baseline comparison
        tuner = RegressionTuner(dimension=dimension, logger=self.logger)
        baseline_emb = self.memory.embedding.get_or_create(self.baseline)
        
        # Precompute baseline embeddings for all contexts
        baseline_inputs = []
        for doc_text, goal_text, _ in absolute_scores:
            goal_emb = self.memory.embedding.get_or_create(goal_text)
            baseline_inputs.append(np.concatenate([goal_emb, baseline_emb]))
        
        # Scale all baseline inputs at once
        baseline_inputs = np.array(baseline_inputs)
        baseline_scaled = scaler.transform(baseline_inputs)
        
        # Process all documents in batches for efficiency
        batch_size = 128
        for i in range(0, len(absolute_scores), batch_size):
            batch_docs = absolute_scores[i:i+batch_size]
            batch_inputs = []
            
            for doc_text, goal_text, _ in batch_docs:
                goal_emb = self.memory.embedding.get_or_create(goal_text)
                doc_emb = self.memory.embedding.get_or_create(doc_text)
                batch_inputs.append(np.concatenate([goal_emb, doc_emb]))
            
            # Scale and convert to tensor
            batch_inputs = np.array(batch_inputs)
            batch_scaled = scaler.transform(batch_inputs)
            batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32)
            baseline_tensor = torch.tensor(baseline_scaled[i:i+batch_size], dtype=torch.float32)
            
            # Get model predictions in batch
            model.eval()
            with torch.no_grad():
                logits = model(batch_tensor, baseline_tensor).cpu().numpy()
            
            # Train tuner with raw logits
            for j, (doc_text, goal_text, abs_score) in enumerate(batch_docs):
                tuner.train_single(float(logits[j]), abs_score)
        
        tuner.save(locator.tuner_file())
        
        # Save metadata
        meta = {
            "dimension": dimension,
            "model_type": "contrastive_ranker",
            "target_type": self.target_type,
            "embedding_type": self.embedding_type,
            "version": self.version,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "baseline": self.baseline,
            "min_score": self.cfg.get("min_score", 0),
            "max_score": self.max_score,
            "dim": self.dim,
            "hdim": self.hdim,
            "training_pairs": len(samples),
            "calibration_samples": len(absolute_scores)
        }
        self._save_meta_file(meta, dimension)
        
        self.log_event("ContrastiveRankerTrainingComplete", {
            "dimension": dimension,
            "pairs": len(samples),
            "calibration_samples": len(absolute_scores)
        })
        
        return meta