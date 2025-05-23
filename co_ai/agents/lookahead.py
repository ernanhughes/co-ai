# co_ai/agents/lookahead.py
import re
from dataclasses import asdict

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, LOOKAHEAD
from co_ai.models import Lookahead


class LookaheadAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict):
        goal = self.memory.goal.get_or_create(context.get(GOAL))

        # Build context for prompt template
        prompt_context = {
            "goal": goal.goal_text,
            **context
        }

        prompt_template = self.prompt_loader.load_prompt(self.cfg, context)

        # Call LLM to generate anticipated issues and fallbacks
        response = self.call_llm(prompt_template, prompt_context).strip()

        # Store the reflection for traceability
        model_name = self.cfg.get("model").get("name")
        extracted = self.parse_response(response)
        context.update(extracted)
        pipeline = context.get("pipeline", [])
        reflection = Lookahead(
            goal=goal.goal_text,
            agent_name=self.name,
            model_name=model_name,
            input_pipeline=pipeline,
            suggested_pipeline=extracted.get("suggested_pipeline"),
            rationale=extracted.get("rationale"),
            reflection=response,
            metadata={"raw_output": response},
            run_id=context.get("run_id"),
        )
        reflection.store(self.memory, self.logger)

        # Log the result
        self.logger.log("LookaheadGenerated", {
            "goal": goal.goal_text,
            "lookahead": response[:250]  # short preview
        })

        # Store in context
        context[self.output_key] = asdict(reflection)
        return context

    def parse_response(self, text: str) -> dict:
        import re

        suggested = re.search(r"# Suggested Pipeline\s*(.*?)\n#", text, re.DOTALL)
        rationale = re.search(r"# Rationale\s*(.*)", text, re.DOTALL)

        pipeline = suggested.group(1).strip().splitlines() if suggested else []
        pipeline = [line.strip("- ").strip() for line in pipeline if line.strip()]

        return {
            "suggested_pipeline": pipeline if pipeline else None,
            "rationale": rationale.group(1).strip() if rationale else None,
        }


    def extract_sections(self, text: str) -> dict:
        # Simple section splitting
        risks_match = re.search(r"# Predicted Risks\s*(.*?)(?:#|$)", text, re.DOTALL)
        backups_match = re.search(r"# Backup Plans\s*(.*)", text, re.DOTALL)

        return {
            "rationale": risks_match.group(1).strip() if risks_match else None,
            "backup_plans": [
                line.strip("- ").strip()
                for line in (backups_match.group(1).strip().split("\n") if backups_match else [])
                if line.strip()
            ]
        }
