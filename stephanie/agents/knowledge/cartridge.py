from stephanie.agents.world.base_agent import BaseAgent
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
from stephanie.models.cartridge import CartridgeORM
from stephanie.analysis.domain_classifier import DomainClassifier

CARTRIDGE_UPDATE_THRESHOLD = 0.7  # Threshold for updating cartridges based on feedback

    
class CartridgeAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.force_domain_update = cfg.get("force_domain_update", False)
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.min_classification_score = cfg.get("min_classification_score", 0.6)
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

                full_text = f"# {title}\n\n## Summary\n\n{summary}\n\n## Content\n\n{text}"


                # Get embedding
                self.memory.embedding.get_or_create(text)
                embedding_vector_id = self.memory.embedding.get_id_for_text(text)
                if not embedding_vector_id:
                    self.logger.log("EmbeddingNotFound", {"text": text[:100]})
                    continue

                sections = self._split_into_sections(text, context)

                # Create and store CartridgeORM
                cartridge = CartridgeORM(
                    goal_id=goal_id,
                    source_type="document",
                    source_uri=str(doc_id),
                    markdown_content=full_text,
                    title=title,
                    summary=summary,
                    sections=sections,
                    triples=[],  # could extract via NLP if available
                    domain_tags=[],  # filled later
                    embedding_id=embedding_vector_id,
                    created_at=datetime.utcnow(),
                )
                self.memory.session.add(cartridge)
                self.memory.session.flush()  # get cartridge.id
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
        merged_context = {"text":text, **context}
        prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)
        response = self.call_llm(prompt, context=merged_context)
        return self.parse_response(response)


    def parse_response(self, markdown_text: str) -> dict:
        """
        Parses the LLM markdown response into structured cartridge components.
        Expected sections:
        - Summary (first paragraph)
        - Key Points (bullet list)
        - Strategic Implications (markdown section)
        - Related Concepts or Domains (bullet list)

        Returns:
            dict with: title, summary, sections, domain_tags
        """
        result = {
            "title": None,
            "summary": "",
            "sections": {},
            "triples": [],
            "domain_tags": []
        }

        # Optional: extract first H1 or H2 as title
        title_match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
        if title_match:
            result["title"] = title_match.group(1).strip()

        # Extract summary (first paragraph, skipping title)
        summary_match = re.search(r"(?<=\n\n)([^#\n][^\n]+(?:\n(?!#).+)*)", markdown_text)
        if summary_match:
            result["summary"] = summary_match.group(1).strip()

        # Extract Key Points (markdown bullets)
        key_points = re.findall(r"(?m)^[-*+]\s+(.*)", markdown_text)
        if key_points:
            result["sections"]["Key Points"] = key_points

        # Extract Strategic Implications section
        strat_imp_match = re.search(r"(?i)##?\s+Strategic Implications\s*\n(.+?)(?=\n##|\Z)", markdown_text, re.DOTALL)
        if strat_imp_match:
            result["sections"]["Strategic Implications"] = strat_imp_match.group(1).strip()

        # Extract Related Concepts or Domains
        related_match = re.search(r"(?i)##?\s+Related Concepts(?: or Domains)?\s*\n(.+?)(?=\n##|\Z)", markdown_text, re.DOTALL)
        if related_match:
            domains = re.findall(r"[-*+]\s+(.*)", related_match.group(1))
            result["domain_tags"] = [d.strip() for d in domains if d.strip()]

        return result
