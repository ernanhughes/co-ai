# stephanie/agents/maintenance/mrq_trainer.py
import json
import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.models.model_version import ModelVersionORM
from stephanie.models.training_stats import TrainingStatsORM
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.model import MRQModel
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.model_locator import ModelLocator


class MRQTrainer:
    class Locator:
        """Integrated model path resolver"""
        def __init__(
            self,
            root_dir="models",
            dimension="alignment",
            embedding_type="hnet",
            model_type="mrq",
            target_type="document",
            version="v1"
        ):
            self.root_dir = root_dir
            self.dimension = dimension
            self.embedding_type = embedding_type
            self.model_type = model_type
            self.target_type = target_type
            self.version = version
            self._validate_inputs()

        def _validate_inputs(self):
            """Ensure valid input types"""
            if not isinstance(self.dimension, str):
                raise ValueError("Dimension must be a string")
            if not isinstance(self.version, str):
                raise ValueError("Version must be a string")
            if self.model_type not in ["mrq", "sicql"]:
                raise ValueError("Invalid model type")

        @property
        def base_path(self) -> str:
            """Build hierarchical path: models/embedding_type/model_type/target_type/dimension/version"""
            path = os.path.join(
                self.root_dir,
                self.embedding_type,
                self.model_type,
                self.target_type,
                self.dimension,
                self.version
            )
            os.makedirs(path, exist_ok=True)
            return path

        def encoder_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_encoder.pt")

        def predictor_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_predictor.pt")

        def meta_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.meta.json")

        def tuner_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.tuner.json")

        def model_exists(self) -> bool:
            """Check if model files exist"""
            return (
                os.path.exists(self.encoder_file()) and 
                os.path.exists(self.predictor_file())
            )

    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training configuration
        self._init_config(cfg)
        
        # Initialize locator
        self.locator = self.Locator(
            root_dir=cfg.get("model_path", "models"),
            dimension=cfg.get("dimension", "alignment"),
            embedding_type=cfg.get("embedding_type", "hnet"),
            model_type="mrq",
            target_type=cfg.get("target_type", "document"),
            version=cfg.get("model_version", "v1")
        )
        
        # Track training state
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.models = {}
        self.tuners = {}
        self._load_tuners()
        
        self.logger.log("MRQTrainerInitialized", {
            "dimension": self.cfg.get("dimension", "alignment"),
            "embedding_type": self.cfg.get("embedding_type", "hnet"),
            "use_tuner": self.use_tuner,
            "device": str(self.device)
        })

    def _init_config(self, cfg):
        """Initialize training parameters from config"""
        self.use_tuner = cfg.get("use_tuner", True)
        self.use_early_stopping = cfg.get("early_stopping", True)
        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.batch_size = cfg.get("batch_size", 32)
        self.epochs = cfg.get("epochs", 50)
        self.lr = cfg.get("lr", 1e-4)
        self.gamma = cfg.get("gamma", 0.95)  # Discount factor
        self.dimensions = cfg.get("dimensions", [])
        self.min_samples = cfg.get("min_samples", 100)

    def _load_tuners(self):
        """Load regression tuners for each dimension"""
        for dim in self.dimensions:
            tuner_path = self.Locator(
                dimension=dim,
                embedding_type=self.cfg.get("embedding_type", "hnet")
            ).tuner_file()
            
            if os.path.exists(tuner_path):
                self.tuners[dim] = RegressionTuner(dimension=dim)
                self.tuners[dim].load(tuner_path)
            else:
                self.tuners[dim] = None
                self.logger.log("TunerMissing", {
                    "dimension": dim,
                    "path": tuner_path
                })

    def _build_model(self, dim):
        """Build or load MRQ model"""
        if self.locator.model_exists():
            # Load existing model
            encoder = TextEncoder(
                dim=self.dim, 
                hdim=self.hdim
            ).to(self.device)
            predictor = ValuePredictor(
                zsa_dim=self.dim, 
                hdim=self.hdim
            ).to(self.device)
            
            encoder.load_state_dict(
                torch.load(self.locator.encoder_file(), map_location=self.device)
            )
            predictor.load_state_dict(
                torch.load(self.locator.predictor_file(), map_location=self.device)
            )
            
            return MRQModel(encoder, predictor, self.memory.embedding, device=self.device)
        
        # Build new model
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim
        encoder = TextEncoder(
            dim=self.dim, 
            hdim=self.hdim
        ).to(self.device)
        predictor = ValuePredictor(
            zsa_dim=self.dim, 
            hdim=self.hdim
        ).to(self.device)
        
        return MRQModel(encoder, predictor, self.memory.embedding, device=self.device)

    def _create_dataloader(self, samples):
        """Convert samples to DataLoader with validation"""
        valid_samples = []
        for s in samples:
            context_text = s.get("title", "")
            doc_text = s.get("output", "")
            
            # Validate text
            if not context_text or not doc_text:
                continue
            
            # Get embeddings
            context_emb = torch.tensor(
                self.memory.embedding.get_or_create(context_text)
            ).to(self.device)
            
            doc_emb = torch.tensor(
                self.memory.embedding.get_or_create(doc_text)
            ).to(self.device)
            
            # Get score
            score = s.get("score", 0.5)
            if not isinstance(score, (float, int)):
                continue
            
            valid_samples.append({
                "context": context_emb,
                "document": doc_emb,
                "score": score
            })
        
        if len(valid_samples) < self.min_samples:
            self.logger.log("InsufficientSamples", {
                "dimension": self.cfg.get("dimension", "alignment"),
                "sample_count": len(valid_samples),
                "threshold": self.min_samples
            })
            return None
        
        # Convert to tensors
        context_tensors = torch.stack([s["context"] for s in valid_samples]).to(self.device)
        doc_tensors = torch.stack([s["document"] for s in valid_samples]).to(self.device)
        scores = torch.tensor([s["score"] for s in valid_samples]).to(self.device)
        
        # Create dataset
        dataset = TensorDataset(context_tensors, doc_tensors, scores)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )

    def _train_epoch(self, model, dataloader):
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        count = 0
        
        for ctx_emb, doc_emb, scores in tqdm(dataloader, desc="Training"):
            # Device management
            ctx_emb = ctx_emb.to(self.device)
            doc_emb = doc_emb.to(self.device)
            scores = scores.to(self.device)
            
            # Forward pass
            zsa = model.encoder(ctx_emb, doc_emb)
            predictions = model.predictor(zsa).squeeze()
            
            # Compute loss
            loss = F.mse_loss(predictions, scores)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item() * ctx_emb.size(0)
            count += ctx_emb.size(0)
        
        avg_loss = total_loss / count
        return avg_loss

    def _should_stop_early(self, current_loss):
        """Check for early stopping"""
        if not self.use_early_stopping:
            return False
        
        if current_loss < self.best_loss - self.early_stopping_min_delta:
            self.best_loss = current_loss
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            return self.early_stop_counter >= self.early_stopping_patience

    def _save_model(self, model, dim, avg_loss):
        """Save model components"""
        # Save encoder and predictor
        torch.save(model.encoder.state_dict(), self.locator.encoder_file())
        torch.save(model.predictor.state_dict(), self.locator.predictor_file())
        
        # Save metadata
        meta = {
            "dim": self.dim,
            "hdim": self.hdim,
            "dimension": dim,
            "version": self.cfg.get("model_version", "v1"),
            "avg_loss": avg_loss,
            "device": str(self.device),
            "embedding_type": self.cfg.get("embedding_type", "hnet"),
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(self.locator.meta_file(), "w") as f:
            json.dump(meta, f)
        
        # Save tuner if available
        if dim in self.tuners and self.tuners[dim]:
            self.tuners[dim].save(self.locator.tuner_file())
        
        # Save model version
        model_version = ModelVersionORM(**meta)
        self.memory.session.add(model_version)
        self.memory.session.commit()
        
        return meta

    def _log_training_stats(self, dim, meta):
        """Log training stats to database"""
        training_stats = TrainingStatsORM(
            model_type="mrq",
            target_type=self.cfg.get("target_type", "document"),
            dimension=dim,
            version=meta["version"],
            avg_q_loss=meta["avg_loss"],
            performance=meta["avg_loss"],
            policy_entropy=meta.get("policy_entropy", 0.0),
            policy_stability=meta.get("policy_stability", 0.0)
        )
        self.memory.session.add(training_stats)
        self.memory.session.commit()

    def train(self, samples, dim=None):
        """
        Train MRQ model for a dimension
        Args:
            samples: List of training samples
            dim: Dimension to train
        Returns:
            Training statistics and model
        """
        dim = dim or self.cfg.get("dimension", "alignment")
        self.logger.log("DimensionTrainingStarted", {"dimension": dim})
        
        # Prepare data
        dataloader = self._create_dataloader(samples)
        if not dataloader:
            return {"error": "insufficient_data", "dimension": dim}
        
        # Build model
        model = self._build_model(dim)
        model.train()
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode="min", 
            factor=0.5, 
            patience=2
        )
        
        # Training stats
        stats = {
            "dimension": dim,
            "losses": [],
            "policy_entropies": [],
            "avg_q_loss": 0.0,
            "policy_entropy": 0.0,
            "policy_stability": 0.0
        }

        # Training loop
        for epoch in range(self.epochs):
            avg_loss = self._train_epoch(model, dataloader)
            stats["losses"].append(avg_loss)
            
            # Early stopping
            if self._should_stop_early(avg_loss):
                self.logger.log("EarlyStoppingTriggered", {
                    "dimension": dim,
                    "epoch": epoch + 1,
                    "best_loss": self.best_loss
                })
                break
                
            # Learning rate scheduling
            if self.use_early_stopping:
                scheduler.step(avg_loss)
        
        # Save model
        meta = self._save_model(model, dim, stats["losses"][-1])
        stats.update(meta)
        
        # Log to database
        self._log_training_stats(dim, meta)
        
        self.logger.log("DimensionTrainingComplete", {
            "dimension": dim,
            "final_loss": stats["avg_q_loss"],
            "sample_count": len(samples)
        })
        
        # Cache model
        self.models[dim] = model
        return stats

    def _log_training_stats(self, dim, meta):
        """Log training stats to database"""
        training_stats = TrainingStatsORM(
            model_type="mrq",
            target_type=self.cfg.get("target_type", "document"),
            dimension=dim,
            version=meta["version"],
            avg_q_loss=meta["avg_loss"],
            policy_entropy=meta.get("policy_entropy", 0.0),
            policy_stability=meta.get("policy_stability", 0.0)
        )
        self.memory.session.add(training_stats)
        self.memory.session.commit()

    def _build_model(self, dim):
        """Build or load MRQ model"""
        if self.locator.model_exists():
            # Load existing model
            encoder = TextEncoder(
                dim=self.dim, 
                hdim=self.hdim
            ).to(self.device)
            
            predictor = ValuePredictor(
                zsa_dim=self.dim, 
                hdim=self.hdim
            ).to(self.device)
            
            encoder.load_state_dict(
                torch.load(self.locator.encoder_file(), map_location=self.device)
            )
            predictor.load_state_dict(
                torch.load(self.locator.predictor_file(), map_location=self.device)
            )
            
            return MRQModel(encoder, predictor, self.memory.embedding, device=self.device)
        
        # Build new model
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim
        
        encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
        predictor = ValuePredictor(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        
        return MRQModel(encoder, predictor, self.memory.embedding, device=self.device)

    def _train_epoch(self, model, dataloader):
        """Train for one epoch"""
        total_loss = 0.0
        count = 0
        
        for ctx_emb, doc_emb, scores in dataloader:
            # Ensure device alignment
            ctx_emb = ctx_emb.to(self.device)
            doc_emb = doc_emb.to(self.device)
            scores = scores.to(self.device)
            
            # Forward pass
            zsa = model.encoder(ctx_emb, doc_emb)
            predictions = model.predictor(zsa).squeeze()
            
            # Compute loss
            loss = F.mse_loss(predictions, scores)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item() * ctx_emb.size(0)
            count += ctx_emb.size(0)
        
        avg_loss = total_loss / count
        self.logger.log("MRQTrainingEpoch", {
            "epoch": self.epoch + 1,
            "loss": avg_loss,
            "lr": self.optimizer.param_groups[0]["lr"]
        })
        
        return avg_loss

    def _should_stop_early(self, current_loss):
        """Early stopping logic"""
        if not self.use_early_stopping:
            return False
        
        if self.epoch == 0:
            self.best_loss = current_loss
            return False
        
        # Check for improvement
        if current_loss < self.best_loss - self.early_stopping_min_delta:
            self.best_loss = current_loss
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
        
        return self.early_stop_counter >= self.early_stopping_patience

    def _save_model(self, model, dim, avg_loss):
        """Save model components with metadata"""
        # Save model files
        torch.save(model.encoder.state_dict(), self.locator.encoder_file())
        torch.save(model.predictor.state_dict(), self.locator.predictor_file())
        
        # Calculate policy entropy (if available)
        policy_entropy = 0.0
        policy_stability = 0.0
        
        if hasattr(model, "pi_head"):
            with torch.no_grad():
                policy_logits = model.pi_head.weight.data.mean(dim=0)
                policy_probs = F.softmax(policy_logits, dim=-1)
                policy_entropy = -torch.sum(
                    policy_probs * torch.log(policy_probs + 1e-8), 
                    dim=-1
                ).mean().item()
                policy_stability = policy_probs.max().item()
        
        # Build metadata
        meta = {
            "dim": self.dim,
            "hdim": self.hdim,
            "dimension": dim,
            "version": self.cfg.get("model_version", "v1"),
            "avg_loss": avg_loss,
            "policy_entropy": policy_entropy,
            "policy_stability": policy_stability,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save metadata
        with open(self.locator.meta_file(), "w") as f:
            json.dump(meta, f)
        
        return meta

    def _load_model(self, dim):
        """Load model with fallback"""
        if dim in self.models:
            return self.models[dim]
            
        if self.locator.model_exists():
            # Load existing model
            encoder = TextEncoder(
                dim=self.dim, 
                hdim=self.hdim
            ).to(self.device)
            
            predictor = ValuePredictor(
                zsa_dim=self.dim, 
                hdim=self.hdim
            ).to(self.device)
            
            encoder.load_state_dict(
                torch.load(self.locator.encoder_file(), map_location=self.device)
            )
            predictor.load_state_dict(
                torch.load(self.locator.predictor_file(), map_location=self.device)
            )
            
            model = MRQModel(encoder, predictor, self.memory.embedding, device=self.device)
            self.models[dim] = model
            return model
        
        # Build new model
        return self._build_model(dim)

    def _build_model(self, dim):
        """Build new MRQ model"""
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim
        
        encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
        predictor = ValuePredictor(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        
        model = MRQModel(encoder, predictor, self.memory.embedding, device=self.device)
        self.models[dim] = model
        return model

    def _train_mrq(self, model, dataloader, output_dir=None):
        """Train standard MRQ model"""
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        # Build optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        
        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0.0
            count = 0
            
            for context_emb, doc_emb, scores in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                # Device management
                context_emb = context_emb.to(self.device)
                doc_emb = doc_emb.to(self.device)
                scores = scores.to(self.device)
                
                # Forward pass
                zsa = model.encoder(context_emb, doc_emb)
                q_pred = model.predictor(zsa).squeeze()
                
                # Compute loss
                loss = F.mse_loss(q_pred, scores)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                # Track loss
                total_loss += loss.item() * context_emb.size(0)
                count += context_emb.size(0)
            
            avg_loss = total_loss / count
            self.logger.log("MRQTrainingEpoch", {
                "epoch": epoch + 1,
                "loss": avg_loss
            })
            
            # Early stopping
            if avg_loss < best_loss - self.early_stopping_min_delta:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                torch.save(model.encoder.state_dict(), f"{output_dir}/encoder.pt")
                torch.save(model.predictor.state_dict(), f"{output_dir}/predictor.pt")
            else:
                patience_counter += 1
            
            # Check for early stopping
            if patience_counter >= self.early_stopping_patience:
                self.logger.log("MRQEarlyStopping", {
                    "epoch": epoch + 1,
                    "best_loss": best_loss
                })
                break
        
        self.logger.log("MRQTrainingComplete", {"best_loss": best_loss})
        return model

    def run(self, context: dict) -> dict:
        """Main entry point for training"""
        documents = context.get("documents", [])
        goal = context.get("goal", {})
        
        # Discover dimensions if not specified
        if not self.dimensions:
            self.dimensions = ModelLocator.discover_dimensions(
                self.cfg.get("model_path", "models"),
                self.cfg.get("embedding_type", "hnet"),
                "mrq",
                self.cfg.get("target_type", "document")
            )
        
        # Train each dimension
        results = {}
        for dim in self.dimensions:
            # Get training samples
            samples = self._get_samples(context, documents, dim)
            if not samples:
                continue
            
            # Train model
            stats = self.train(samples, dim)
            if "error" in stats:
                continue
            
            # Update belief cartridges
            self._update_belief_cartridge(context, dim, stats)
            results[dim] = stats
        
        # Update context with results
        context["training_stats"] = results
        return context

    def _get_samples(self, context,  documents, dim):
        """Get training samples for dimension"""
        samples = []
        goal = context.get("goal", {})
        for doc in documents:
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            score = self.memory.scores.get_score(goal_id=goal.id, scorable_id=scorable.id)
            if score:
                samples.append({
                    "title": goal.get("goal_text", ""),
                    "output": scorable.text,
                    "score": score.score
                })
        return samples

    def _update_belief_cartridge(self, context, dim, stats):
        """Update belief cartridges with training stats"""
        if not stats.get("policy_entropy"):
            policy_logits = [0.3, 0.7, 0.0]  # Default policy
        else:
            policy_logits = stats["policy_logits"]
        
        # Build belief cartridge
        cartridge = BeliefCartridgeORM(
            title=f"{dim} policy",
            content=f"Policy logits: {policy_logits}",
            goal_id=context.get("goal_id"),
            domain=dim,
            policy_logits=policy_logits,
            policy_entropy=stats.get("policy_entropy", 0.0),
            policy_stability=stats.get("policy_stability", 0.0)
        )
        self.memory.session.add(cartridge)
        self.memory.session.commit()

    def _calculate_policy_logits(self, model):
        """Calculate policy logits from encoder weights"""
        if not hasattr(model, "pi_head"):
            return [0.3, 0.7, 0.0]  # Default
        
        with torch.no_grad():
            policy_weights = model.pi_head.weight.data.mean(dim=0)
            policy_probs = F.softmax(policy_weights, dim=-1)
            return policy_probs.tolist()

    def _calculate_policy_stability(self, policy_logits):
        """Calculate policy stability from logits"""
        if not policy_logits:
            return 0.0
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1)
        return policy_probs.max().item()

    def _calculate_policy_entropy(self, policy_logits):
        """Calculate policy entropy"""
        if not policy_logits:
            return 0.0
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1)
        entropy = -torch.sum(
            policy_probs * torch.log(policy_probs + 1e-8), 
            dim=-1
        ).mean().item()
        return entropy

    def _validate_tensor(self, tensor, name):
        """Validate tensor before use"""
        if tensor is None:
            self.logger.log("InvalidTensor", {
                "tensor_name": name,
                "reason": "tensor_is_none"
            })
            return False
        
        if torch.isnan(tensor).any():
            self.logger.log("NaNInTensor", {
                "tensor_name": name,
                "tensor": tensor.tolist()
            })
            return False
        
        return True

    def _save_to_db(self, dim, stats):
        """Save training stats to database"""
        training_stat = TrainingStatsORM(
            model_type="mrq",
            target_type=self.cfg.get("target_type", "document"),
            dimension=dim,
            version=self.cfg.get("model_version", "v1"),
            avg_q_loss=stats["avg_q_loss"],
            policy_entropy=stats.get("policy_entropy", 0.0),
            policy_stability=stats.get("policy_stability", 0.0)
        )
        self.memory.session.add(training_stat)
        self.memory.session.commit()