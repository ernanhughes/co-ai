from co_ai.agents import BaseAgent
import litellm
from co_ai.constants import (
    GOAL,
    REFLECTION,
    HYPOTHESES,
    MODEL,
    API_BASE,
    API_KEY,
    STRATEGY,
    PROMPT_PATH,
    SAVE_PROMPT
)
from co_ai.memory.embedding_store import is_duplicate_embedding, store_sample
import random
import uuid
import numpy as np


class GRAAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.generator = cfg.get("generator_pool")
        self.adjudicator_pool = cfg.get("adjudicator_pool")
        self.score_threshold = cfg.get("score_threshold")
        self.std_threshold = cfg.get("std_threshold")
        self.dedup_similarity = cfg.get("dedup_similarity")
        self.samples_per_iteration = cfg.get("samples_per_iteration")
        self.iterations = cfg.get("iterations")

    def run(self, context: dict) -> dict:
        for _ in range(self.iterations):
            for _ in range(self.samples_per_iteration):
                self.generate_and_process(context)
        return context

    def generate_and_process(self, context: dict):
        generator = random.choice.models.generator_pool
        keywords = self.generate_keywords()
        summary = self.generate_summary(keywords)

        instruction, response = self.generate_instruction_response(
            generator, keywords, summary
        )
        reviews = self.run_reviewers(instruction, response)

        mean_score = np.mean([r["score"] for r in reviews])
        std_dev = np.std([r["score"] for r in reviews])

        if mean_score >= self.cfg.thresholds.score_threshold:
            if std_dev <= self.cfg.thresholds.std_threshold:
                accepted = True
            else:
                adjudicator = random.choice(self.cfg.models.adjudicator_pool)
                accepted = self.run_adjudicator(
                    adjudicator, instruction, response, reviews
                )
        else:
            accepted = False

        if accepted:
            emb = self.memory.embedding.get_or_create(instruction + response)
            if not is_duplicate_embedding(
                emb, threshold=self.dedup_similarity
            ):
                store_sample(instruction, response, emb)
                self.logger.log(
                    "accepted_sample",
                    {"instruction": instruction, "response": response},
                )

    def generate_instruction_response(self, model, keywords, summary, context: dict):
        prompt = f"Using keywords: {', '.join(keywords)} and this summary:\n'{summary}', generate an instruction and response."
        result = self.call_ollama(model, prompt)
        return self.split_instruction_response(result)

    def split_instruction_response(self, output):
        if "## Instruction:" in output and "## Response:" in output:
            instruction = (
                output.split("## Instruction:")[1].split("## Response:")[0].strip()
            )
            response = output.split("## Response:")[1].strip()
        else:
            instruction, response = output.strip().split("\n", 1)
        return instruction, response

    def run_reviewers(self, instruction, response, context: dict):
        scores = []
        for reviewer in self.cfg.models.reviewer_pool:
            prompt = "Score the response to the instruction on 6 dimensions (0–10 each), then average them."
            full = f"Instruction: {instruction}\nResponse: {response}\n\n{prompt}"
            result = self.call_ollama(reviewer, full)
            scores.append(
                {
                    "reviewer": reviewer,
                    "score": self.parse_score(result),
                    "comment": result,
                }
            )
        return scores

    def run_adjudicator(self, model, instruction, response, reviews, context: dict):
        combined_feedback = "\n".join([r["comment"] for r in reviews])
        prompt = f"As an adjudicator, reassess this response given the following reviewer feedback:\n{combined_feedback}"
        result = self.call_ollama(model, prompt)
        score = self.parse_score(result)
        return score >= self.cfg.thresholds.score_threshold

    def parse_score(self, text):
        import re

        match = re.search(r"(\d+(\.\d+)?)", text)
        return float(match.group(1)) if match else 0.0

    def generate_keywords(self):
        return random.sample(
            ["Fantasy", "Logic", "Code", "Philosophy", "Economy", "Ethics"], 3
        )

    def generate_summary(self, keywords):
        return f"A task involving {', '.join(keywords)}"


    def call_ollama(self, model, prompt, context):
        messages = [{"role": "user", "content": prompt}]
        try:
            response = litellm.completion(
                model=model.get(MODEL),
                messages=messages,
                api_base=model.get(API_BASE),
                api_key=model.get(API_KEY),
            )
            output = response["choices"][0]["message"]["content"]
            if self.cfg.get(SAVE_PROMPT, False) and self.memory:
                self.memory.prompt.save(
                    context.get("goal"),
                    agent_name=self.name,
                    prompt_key=self.cfg.get(PROMPT_PATH, ""),
                    prompt_text=prompt,
                    response=output,
                    # source=self.cfg.get("prompt_type", "file"),
                    strategy=self.cfg.get(STRATEGY, ""),
                    version=self.cfg.get("version", 1),
                    # metadata={}
                )
            response = self.remove_think_blocks(output)
            if self.cfg.get("add_prompt_to_history", True):
                self.add_to_prompt_history(context, prompt, {"response":response})
            return response
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log("LLMCallError", {"exception": e})
            raise
