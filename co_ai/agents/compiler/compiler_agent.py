# co_ai/compiler/compiler_agent.py
from co_ai.compiler import (
    reasoning_trace, node_executor, scorer, symbol_mapper,
    step_selector, final_prompt_builder
)
from co_ai.agents.base_agent import BaseAgent

class CompilerAgent(BaseAgent):
    def __init__(self, cfg, agent: BaseAgent, logger=None):
        super().__init__(cfg, logger=logger)
        self.tree = reasoning_trace.ReasoningTree()
        self.executor = node_executor.NodeExecutor(agent)
        self.scorer = scorer.ReasoningNodeScorer("mrq")
        self.mapper = symbol_mapper.SymbolMapper()
        self.selector = step_selector.StepSelector()
        self.builder = final_prompt_builder.FinalPromptBuilder()

    async def run(self, goal: str) -> str:
        root_id = self.tree.add_root(goal, "Start solving this problem.", "Generate initial plan")
        node = self.tree.nodes[root_id]

        for iteration in range(5):  # Max iterations
            result = await self.executor.execute(node)
            score = self.scorer.score(node)
            node.response = result["response"]
            node.score = score
            self.mapper.tag_node(node)

            next_steps = self.selector.select_next_steps(self.tree)
            for step in next_steps:
                child_id = self.tree.add_child(
                    parent_id=node.id,
                    thought=step["thought"],
                    action=step["action"],
                    response="",
                    score=0.0
                )
                node = self.tree.nodes[child_id]

        best_path = self.tree.get_best_path()
        final_prompt = self.builder.build_prompt(best_path)
        return final_prompt