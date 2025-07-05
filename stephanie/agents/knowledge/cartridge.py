from stephanie.agents.world.base_agent import BaseAgent
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
from stephanie.models.cartridge import CartridgeORM
from stephanie.analysis.domain_classifier import DomainClassifier
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.models.evaluation import TargetType
from stephanie.scoring.scorable import Scorable

CARTRIDGE_UPDATE_THRESHOLD = 0.7  # Threshold for updating cartridges based on feedback


class CartridgeAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.force_domain_update = cfg.get("force_domain_update", False)
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.min_classification_score = cfg.get("min_classification_score", 0.6)
        self.triplets_file = cfg.get("triplets_file", "triplet.txt")
        self.domain_classifier = DomainClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=cfg.get("domain_seed_config_path", "config/domain/cartridges.yaml"),
        )

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        cartridges = []
        for doc in documents:
            try:
                doc_id = doc["id"]
                title = doc.get("title", f"Document {doc_id}")
                summary = doc.get("summary", "")
                text = doc.get("content", doc.get("text", ""))
                goal_id = context.get("goal_id")  # optional

                # full_text = f"# {title}\n\n## Summary\n\n{summary}\n\n## Content\n\n{text}"

                self.memory.embedding.get_or_create(text)
                embedding_vector_id = self.memory.embedding.get_id_for_text(text)
                if not embedding_vector_id:
                    self.logger.log("EmbeddingNotFound", {"text": text[:100]})
                    continue

                sections = self._split_into_sections(text, context)
                triplets = self.construct_triplets(sections, context)

                cartridge = CartridgeORM(
                    goal_id=goal_id,
                    source_type="document",
                    source_uri=str(doc_id),
                    # markdown_content=full_text,
                    title=title,
                    summary=summary,
                    sections=sections,
                    triples=[],
                    domain_tags=[],
                    embedding_id=embedding_vector_id,
                    created_at=datetime.utcnow(),
                )
                cartridge.markdown_content = self.format_cartridge_text(cartridge)
                cartridge = self.memory.cartridges.add_cartridge(cartridge.to_dict())
                merged_context = {
                    "cartridge": cartridge.to_dict(),
                    "goal": context.get("goal"),
                    **context
                }
                scorable = Scorable(id=cartridge.id, text=cartridge.markdown_content, target_type=TargetType.CARTRIDGE)
                score = self.score_item(scorable, merged_context, metrics="cartridge")
                self.logger.log("CartridgeScored", score.to_dict())
                context.setdefault("cartridge_scores", []).append(score.to_dict())

                for subj, pred, obj in triplets:
                    if not subj or not pred or not obj:
                        self.logger.log("InvalidTriplet", {
                            "subject": subj, "predicate": pred, "object": obj
                        })
                        continue
                    self.memory.cartridge_triples.insert({
                        "cartridge_id": cartridge.id,
                        "subject": subj,
                        "predicate": pred,
                        "object": obj,
                    })

                self.assign_domains_to_cartridge(cartridge)
                self.logger.log("CartridgeCreated", cartridge.to_dict())
                cartridges.append(cartridge.to_dict())
            except Exception as e:
                self.logger.log(
                    "CartridgeCreateLoadFailed", {"id": doc.get("id"), "error": str(e)}
                )

        context[self.output_key] = cartridges
        context["cartridge_ids"] = [cartridge.get("id") for cartridge in cartridges]
        return context

    def assign_domains_to_cartridge(self, cartridge: CartridgeORM): 
        """
        Classifies the document text into one or more domains,
        and stores results in the document_domains table.
        """
        content = cartridge.markdown_content
        if content:
            results = self.domain_classifier.classify(content, self.top_k_domains, self.min_classification_score)
            for domain, score in results:
                self.memory.cartridge_domains.insert({
                    "cartridge_id": cartridge.id,
                    "domain": domain,
                    "score": score,
                })
                self.logger.log("DomainAssigned", {
                    "title": cartridge.title[:60] if cartridge.title else "",
                    "domain": domain,
                    "score": score,
                })
        else:
            self.logger.log("DocumentNoContent", {
                "document_id": cartridge.id,
                "title": cartridge.title[:60] if cartridge.title else "", })

    def _split_into_sections(self, text: str, context:dict) -> dict:
        """
        Splits the text into sections based on common headings.
        This is a simple heuristic and can be improved with NLP.
        """
        merged_context = {"text":text, 
                          "goal": context.get("goal"),
                          **context}
        prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)
        response = self.call_llm(prompt, context=merged_context)
        return self.parse_response(response)

    def construct_triplets(self, points: list, context: dict) -> list[tuple[str, str, str]]:
        merged_context = {"points": points, "goal": context.get("goal"), **context}
        prompt = self.prompt_loader.from_file(self.triplets_file, self.cfg, merged_context)
        response = self.call_llm(prompt, context=merged_context)
        print("Triplet response:\n", response)
        return self.parse_triplets(response)

    def parse_triplets(self, markdown_text: str) -> list[tuple[str, str, str]]:
        pattern = re.compile(r"-\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*(.+?)\s*\)")
        matches = pattern.findall(markdown_text)
        return [(subj.strip(), pred.strip(), obj.strip()) for subj, pred, obj in matches]

    def parse_response(self, markdown_text: str) -> list:
        """
        Extracts bullet-like lines from markdown supporting prefixes like:
        - '- '
        - '* '
        - '** '
        - '# '
        - '## '
        
        Strips common markdown formatting and whitespace.
        """
        # Match lines that start with one or two of: '*', '-', or '#' followed by whitespace
        bullet_pattern = re.compile(r"^\s*([#*-]{1,2})\s+(.*)", re.MULTILINE)
        raw_lines = bullet_pattern.findall(markdown_text)
        bullet_points = [re.sub(r"\*\*(.*?)\*\*", r"\1", content).strip() for _, content in raw_lines]
        return bullet_points

    def format_cartridge_text(self, cartridge: CartridgeORM) -> str:
        """
        Constructs a clean, unified text string from a cartridge's components:
        Title, Summary, and Section Points.

        Useful for scoring, embedding, or logging.
        """
        lines = []

        if cartridge.title:
            lines.append(f"Title: {cartridge.title}")

        if cartridge.summary:
            lines.append(f"Summary: {cartridge.summary}")

        if cartridge.sections:
            for section in cartridge.sections:
                lines.append(f"{section}")

        return "\n\n".join(lines).strip()
