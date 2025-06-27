from co_ai.agents.base_agent import BaseAgent
from co_ai.utils.timing import time_function

class PupilAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    @time_function(logger=None)
    async def run(self, context: dict) -> dict:
        self.execute_prompt(context)
        return context
        
