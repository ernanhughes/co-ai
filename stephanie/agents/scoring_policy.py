from stephanie.agents.base_agent import BaseAgent


class ScoringPolicyAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.ebt = EBTIn
        self.mrq = mrq_agent
        self.llm = llm_agent

    async def run(self, context: dict) -> dict:
        # Implement agent logic here
        return context