# co_ai/compiler/node_executor.py
from co_ai.agents.base_agent import BaseAgent
from co_ai.compiler.reasoning_trace import ReasoningNode

class NodeExecutor:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def execute(self, node: ReasoningNode) -> dict:
        # Run agent on node's action/thought
        context = {
            "goal": node.goal,
            "current_thought": node.thought,
            "previous_actions": self._get_history(node),
        }
        try:
            response = await self.agent.run(context)
            return {
                "response": response,
                "success": True
            }
        except Exception as e:
            return {
                "response": str(e),
                "success": False
            }

    def _get_history(self, node: ReasoningNode) -> list[dict]:
        # Traverse up the tree to collect history
        pass