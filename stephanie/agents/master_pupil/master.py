# stephanie/agents/master_pupil/master.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.utils.timing import time_function


class MasterAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        question = context.get(
            self.input_key, context.get("goal", {}).get("goal_text", "")
        )
        answer = self.answer(question, context)
        context.setdefault(self.output_key, []).append(answer)
        self.logger.log("MasterAnswerGenerated", f"Answered: {answer[:50]}...")
        return context

    @time_function(logger=None)
    def answer(self, question, context):
        return self.call_llm(question, context, self.cfg.get("pupil_model"))
