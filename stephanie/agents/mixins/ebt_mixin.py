# stephanie/agents/mixins/ebt_mixin.py
import os
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from typing import Dict, List, Optional, Union

from stephanie.utils.model_utils import get_model_path, discover_saved_dimensions
from stephanie.utils.file_utils import load_json
from stephanie.scoring.model.ebt_model import EBTModel


class EBTMixin:
    """
    Mixin that provides EBT-based verification, refinement, and scoring.
    Can be added to any class that has:
    - self.cfg (configuration)
    - self.memory (with embedding store)
    - self.logger (optional)
    """
    
    def __init__(self, ebt_cfg: Dict):
        """
        Initialize EBT functionality from config
        
        Args:
            ebt_cfg: Configuration dict with:
                - model_path: Base path for saved models
                - model_type: Type of EBT model (default: "ebt")
                - target_type: Type of target (default: "document")
                - model_version: Model version (default: "v1")
                - dimensions: List of dimensions to load
                - device: Computation device ("cuda" or "cpu")
                - uncertainty_threshold: Energy threshold for uncertainty
        """
        # Configuration
        self.ebt_cfg = ebt_cfg
        self.model_path = ebt_cfg.get("model_path", "models")
        self.model_type = ebt_cfg.get("model_type", "ebt")
        self.target_type = ebt_cfg.get("target_type", "document")
        self.model_version = ebt_cfg.get("model_version", "v1")
        self.dimensions = ebt_cfg.get("dimensions", [])
        self.uncertainty_threshold = ebt_cfg.get("uncertainty_threshold", 0.75)
        
        # Device
        self.device = torch.device(
            ebt_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Model storage
        self.ebt_models: Dict[str, nn.Module] = {}
        self.ebt_meta: Dict[str, Dict] = {}
        
        # Initialize models
        self._initialize_ebt_models()
    
    def _initialize_ebt_models(self):
        """Load EBT models and metadata"""
        if not self.dimensions:
            self.dimensions = discover_saved_dimensions(
                model_type=self.model_type,
                target_type=self.target_type,
                base_path=self.model_path
            )
        
        for dim in self.dimensions:
            model_path = get_model_path(
                self.model_path,
                self.model_type,
                self.target_type,
                dim,
                self.model_version
            )
            infer_path = os.path.join(model_path, f"{dim}.pt")
            meta_path = os.path.join(model_path, f"{dim}.meta.json")
            
            try:
                # Load model
                model = self._load_ebt_model(infer_path)
                self.ebt_models[dim] = model
                
                # Load metadata
                if os.path.exists(meta_path):
                    self.ebt_meta[dim] = load_json(meta_path)
                else:
                    self.ebt_meta[dim] = {"min": 40, "max": 100}
                    
            except Exception as e:
                if self.ebt_cfg.get("strict_load", True):
                    raise RuntimeError(f"Failed to load EBT model for {dim}: {e}")
                else:
                    print(f"Skipping EBT model for {dim} due to error: {e}")
    
    def _load_ebt_model(self, path: str) -> nn.Module:
        """Load EBT model from disk"""
        model = EBTModel().to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model
    
    def get_energy(self, goal: str, text: str, dimension: Optional[str] = None) -> float:
        """Get raw energy value for a document-goal pair"""
        target_dim = dimension or next(iter(self.ebt_models.keys()), None)
        if not target_dim:
            raise ValueError("No EBT models available")
            Problem right I don't see an option here to do research what I would like to do is look at the latest research on papers including the paper that we're working on that follow this car in the process so maybe we can learn from them and integrate parts of it into this solution before we start working on the solution Let's review each paper get the key points so try to understand like what it adds and say if we were adding it into our system so we're going to say well if we wanted this paper what would how would we do it how would we add it to the system And like what does a dad what are the key ideas then this think in terms of our system in the context of our system what these ideas actually fit in or would they add anything or more importantly are they already covered like are they adding anything new or anything novel specifically if that's going to enhance quality if it isn't going to enhance quality we're rejected immediately obviously And what we want to do is what we're looking for out of all of these papers is maybe an agent or an idea that we put into our pipeline our process or maybe it has some kind of new tool to run code and check code some idea like this that would make sense and could add to our pipeline If there is any of these things we'll take it out of the paper we'll put it into our pipeline for review or or we'll just take out the steps out of these papers and put them in the steps in our process towards our solution Being on this test Gen stuff OK one thing I've always hated about software is that as it grows it starts to perform worse And let's make sure that in our agent that the pass failometric is performance so no code is going to go into the age endeavor it's going to reduce performance Umm if there's cold that's going to produce performance So while his performance performance is the speed of operation one the quality of operation two the effectiveness of operation 3 So we have to have I believe sort of pass fail metrics in this process like for instance the code as is it's going to fail all of these tests right which is why we're starting this process now But the colon code as is functional to some degree right I'm not sure what I'm trying to say here but we want to measure just like with multidimensional scoring right We want a multi dimensional approach to the evaluation of code and we always want this grid or this table to be really obvious when we're having the code so let's let's let's let's step through a process here Umm we decide that we need to generate test cases for whatever a module in our code we generate the test cases we run them and whatever It's probably a bad idea Let's go back to your original idea I want to make sure that Dai as in Stephanie and us are very aware of how these chord changes impact the application so for instance for instance let's say we We won't miss this there's no way we'd miss this because this would be tracked within the first commit but after 2 or 3 commits we'd start to realize there's a pattern has emerged here So quite often in in very large projects that I've worked on someone makes a change and it doesn't flow into the system until six or seven months later because no one can really see it it doesn't it makes it into production which without it ever being understood what happens and then you have to do this big git bisect to try and find out what change made the issue Do you understand what I'm saying I want to build a system that this could never possibly happen to because he understands itself it understands these changes and he would see that this change is going to affect it in a negative way here because of XY and Z because it's measured every single thing and it knows how everything falls together It symbolizes itself completely It understands itself completely so when you make a change here at some part here it's going to know that this is going to affect performance here And well when I talk about performance I'm talking about multi dimensional view on the application that covers things like how fast it runs how well it runs what kind of dependencies it has Will it run on multiple operating systems will it actually run against multiple databases will it run without a database will it Will it run like it used to run are we removing functionality here or are we adding unneeded functionality So so to Support this the system will have to have a sort of set of guidelines what it wants like it'll want to conform to Python if you want to conform to certain rules you want to conform to say for instance that a pip install and it runs it launches it'll want to conform to this rule and that will work in that will work in Collab that will work in Windows work and Linux that will work on a Raspberry Pi so it will want to conform to say this rule right This would be this would be a pass fail rule on any code that's added to the system right something like this these these set of core rules you know you know what I'm saying this this is where it starts and how it understands this is is through this almost a linker view it's able to link itself through it's cold it's able to understand what its code does is able to understand why it's set up this way and why the code is that And most importantly when a change happens or when it changes is suggested it will understand that this change is no good so that when it communicates with another LLM the LLM is suggesting this file it's going to come back and say no this won't work because of XY and Z please suggest me a solution that would fit into a framework that does this Now how this works and I have covered this before is this component based approach Everything in the system is going to start going towards small components that do centralized work so we're going to work in a set of components so for example Mr Q will be a component the whole team The model loading saving will be another component the EBT will be another component the SVM will be another component these will be little blobs Now they exist as packages within the system but essentially to Stephanie they'll appear like A set of components Then when we make changes that are going to affect these components that should be centralized typically are localized into the component then hopefully there won't be these kind of changes that break the whole system Hello Yeah Yeah All right now here's the one thing OK now I'm going to the thing is the dream what we have and what we're building right isn't there yet so this is our goal do you know what I mean so we we build Stephanie right now and the reason that this coding has become important is that she's gotten big right and we need to streamline it towards this ideal this goal you'll see in our other chats that we're starting to go more and more towards isolating functionality into components then reusing it across other components in this calm approach this component object model approach this this I believe is the key to successful large scale system implementation OK jack's up
        ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal)).to(self.device)
        doc_emb = torch.tensor(self.memory.embedding.get_or_create(text)).to(self.device)
        
        with torch.no_grad():
            energy = self.ebt_models[target_dim](ctx_emb, doc_emb).squeeze().cpu().item()
        
        return energy
    
    def optimize(self, goal: str, text: str, dimension: Optional[str] = None,
                 steps: int = 10, step_size: float = 0.05) -> Dict:
        """
        Optimize document text by minimizing energy through gradient descent
        
        Returns:
            dict: {
                "refined_text": str,
                "final_energy": float,
                "energy_trace": List[float],
                "converged": bool
            }
        """
        target_dim = dimension or next(iter(self.ebt_models.keys()), None)
        if not target_dim:
            raise ValueError("No EBT models available")
            
        model = self.ebt_models[target_dim]
        ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal)).to(self.device)
        doc_emb = torch.tensor(self.memory.embedding.get_or_create(text)).to(self.device)
        
        # Make document embedding differentiable
        doc_tensor = doc_emb.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([doc_tensor], lr=step_size)
        
        energy_trace = []
        for step in range(steps):
            optimizer.zero_grad()
            energy = model(ctx_emb, doc_tensor)
            energy.backward()
            optimizer.step()
            energy_trace.append(energy.item())
        
        refined_emb = doc_tensor.detach()
        refined_text = self._embedding_to_text(refined_emb, goal, text)
        
        return {
            "refined_text": refined_text,
            "final_energy": energy_trace[-1],
            "energy_trace": [round(e, 4) for e in energy_trace],
            "converged": abs(energy_trace[-1] - energy_trace[0]) < 0.05,
            "dimension": target_dim
        }
    
    def is_unstable(self, goal: str, text: str, dimension: Optional[str] = None) -> bool:
        """Check if prediction is uncertain using energy + gradient magnitude"""
        energy = self.get_energy(goal, text, dimension)
        ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal)).to(self.device)
        doc_emb = torch.tensor(self.memory.embedding.get_or_create(text)).to(self.device)
        
        doc_tensor = doc_emb.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            energy = self.ebt_models[dimension or next(iter(self.ebt_models))](ctx_emb, doc_tensor)
            grad = torch.autograd.grad(energy, doc_tensor, retain_graph=False)[0]
        
        grad_norm = torch.norm(grad).item()
        uncertainty_score = abs(energy.item()) + grad_norm
        
        return uncertainty_score > self.uncertainty_threshold
    
    def score_document(self, goal: str, text: str, dimension: Optional[str] = None) -> float:
        """Get final scaled score from EBT model"""
        energy = self.get_energy(goal, text, dimension)
        meta = self.ebt_meta.get(dimension or next(iter(self.ebt_models)), {"min": 40, "max": 100})
        
        normalized = sigmoid(torch.tensor(energy)).item()
        final_score = normalized * (meta["max"] - meta["min"]) + meta["min"]
        return round(final_score, 4)
    
    def _embedding_to_text(self, embedding, goal, original_text):
        """Convert refined embedding back to text"""
        if self.ebt_cfg.get("use_llm_refinement", False):
            from stephanie.agents.llm import LLMGenerator
            llm = LLMGenerator(self.ebt_cfg.get("llm_config", {}))
            prompt = f"""Improve the following text to better align with this goal:

Goal: {goal}
Original Text: {original_text}

Refine it while preserving content."""
            return llm.generate(prompt)
        
        # Fallback to nearest neighbor search
        return self.memory.embedding.find_closest(embedding, k=1)[0].text
    
    def get_all_scores(self, goal: str, text: str) -> Dict[str, float]:
        """Get scores across all dimensions"""
        scores = {}
        for dim in self.dimensions:
            scores[dim] = self.score_document(goal, text, dim)
        return scores