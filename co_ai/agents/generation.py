# co_ai/agents/generation.py

from co_ai.agents.base import BaseAgent
from co_ai.constants import FEEDBACK, GOAL, HYPOTHESES, LITERATURE, PIPELINE
from co_ai.models import Hypothesis
from co_ai.parsers import extract_hypotheses


class GenerationAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = self.extract_goal_text(context.get(GOAL))

        self.logger.log("GenerationStart", {GOAL: goal})

        # Load literature if available
        literature = context.get(LITERATURE, {})

        # Build context for prompt
        render_context = {
            LITERATURE: literature,
            FEEDBACK: context.get(FEEDBACK, {}),
            HYPOTHESES: context.get(HYPOTHESES, []),
        }

        # Load prompt based on strategy
        prompt = self.prompt_loader.load_prompt(
            self.cfg, context={**context, **render_context}
        )
        response = self.call_llm(prompt, context)

        # Extract hypotheses
        hypotheses = extract_hypotheses(response)
        for h in hypotheses:
            hyp = Hypothesis(
                goal=goal,
                text=h,
                prompt=prompt,
                pipeline_signature=context.get(PIPELINE),
            )
            self.memory.hypotheses.insert(hyp)

        # Update context with new hypotheses
        context[self.output_key] = hypotheses

        # Log event
        self.logger.log(
            "GeneratedHypotheses",
            {
                GOAL: goal,
                HYPOTHESES: hypotheses,
                "prompt_snippet": prompt[:300],
                "response_snippet": response[:500],
            },
        )

        return context
