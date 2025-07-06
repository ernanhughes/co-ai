# stephanie/agents/cartridge_agent.py

from stephanie.agents.world.base_agent import BaseAgent
from stephanie.builders.cartridge_builder import CartridgeBuilder
from stephanie.builders.triplet_extractor import TripletExtractor
from stephanie.scoring.cartridge_scorer import CartridgeScorer
from stephanie.scoring.triplet_scorer import TripletScorer
from stephanie.analysis.domain_classifier import DomainClassifier
from stephanie.models.cartridge import CartridgeORM
from stephanie.agents.mixins.scoring_mixin import ScoringMixin


class CartridgeAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        self.input_key = cfg.get("input_key", "documents")
        self.score_cartridges = cfg.get("score_cartridges", True)
        self.score_triplets = cfg.get("score_triplets", True)
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.min_classification_score = cfg.get("min_classification_score", 0.6)

        self.domain_classifier = DomainClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=cfg.get("domain_seed_config_path", "config/domain/cartridges.yaml"),
        )

        self.builder = CartridgeBuilder(cfg, memory=self.memory, prompt_loader=self.prompt_loader, logger=self.logger, call_llm=self.call_llm)
        self.triplet_extractor = TripletExtractor(cfg=cfg, prompt_loader=self.prompt_loader,  memory=self.memory, logger=self.logger, call_llm=self.call_llm)
        self.cartridge_scorer = CartridgeScorer(scorer=self, logger=self.logger)
        self.triplet_scorer = TripletScorer(scorer=self, logger=self.logger)

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        cartridges = []

        for doc in documents:
            try:
                goal = context.get("goal")

                # 1. Build CartridgeORM
                cartridge = self.builder.build(doc, goal=goal)
                if not cartridge:
                    continue

                # 2. Extract and insert triplets
                if self.memory.cartridge_triples.has_triples(cartridge.id):
                    self.logger.log("TriplesAlreadyExist", {"cartridge_id": cartridge.id})
                    continue

                triplets = self.triplet_extractor.extract(cartridge.sections, context)
                for subj, pred, obj in triplets:
                    triple_orm = self.memory.cartridge_triples.insert({
                        "cartridge_id": cartridge.id,
                        "subject": subj,
                        "predicate": pred,
                        "object": obj,
                    })
                    if self.score_triplets:
                        score = self.triplet_scorer.score_triplet(triple_orm, goal, context)
                        context.setdefault("cartridge_scores", []).append(score)

                 # 3. Extract and insert theorems
                theorems = self.theorem_extractor.extract(cartridge.sections, context)
                for theorem in theorems:
                    theorem.embedding_id = self.memory.embedding.create(theorem.statement)
                    theorem.cartridges.append(cartridge)
                    self.memory.session.add(theorem)

                    # Optional: Score theorem immediately
                    theorem_score = self.theorem_scorer.score_theorem(theorem, goal, context)
                    context.setdefault("theorem_scores", []).append(theorem_score)

                # Commit new theorems
                self.memory.session.commit()

                # 4. Score Cartridge
                if self.score_cartridges:
                    score = self.cartridge_scorer.score_cartridge(cartridge, goal, context)
                    context.setdefault("cartridge_scores", []).append(score)

                # 5. Assign Domains
                self.assign_domains(cartridge)

                self.logger.log("CartridgeCreated", cartridge.to_dict())
                cartridges.append(cartridge.to_dict())

            except Exception as e:
                self.logger.log("CartridgeCreateLoadFailed", {
                    "id": doc.get("id"),
                    "error": str(e)
                })

        context[self.output_key] = cartridges
        context["cartridge_ids"] = [c.get("id") for c in cartridges]
        return context

    def assign_domains(self, cartridge: CartridgeORM):
        """Classify and log domains for the cartridge."""
        if not cartridge.markdown_content:
            return
        results = self.domain_classifier.classify(
            cartridge.markdown_content,
            top_k=self.top_k_domains,
            threshold=self.min_classification_score
        )
        for domain, score in results:
            self.memory.cartridge_domains.insert({
                "cartridge_id": cartridge.id,
                "domain": domain,
                "score": score,
            })
            self.logger.log("DomainAssigned", {
                "title": cartridge.title[:60],
                "domain": domain,
                "score": score
            })
