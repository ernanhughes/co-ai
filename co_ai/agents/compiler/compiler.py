# co_ai/compiler/compiler_agent.py
from co_ai.agents.compiler.reasoning_trace import ReasoningTree
from co_ai.agents.compiler.node_executor import NodeExecutor
from co_ai.agents.compiler.final_prompt_builder import FinalPromptBuilder
from co_ai.agents.compiler.step_selector import StepSelector
from co_ai.agents.compiler.symbol_mapper import SymbolMapper
from co_ai.agents.compiler.scorer import ReasoningNodeScorer
from co_ai.agents.base_agent import BaseAgent
from co_ai.agents.pipeline.pipeline_runner import PipelineRunnerAgent
from co_ai.agents.mixins.scoring_mixin import ScoringMixin

class CompilerAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None, full_cfg=None):
        super().__init__(cfg, memory, logger)
        self.tree = ReasoningTree()
        runner = PipelineRunnerAgent(cfg, memory=memory, logger=logger, full_cfg=cfg)
        self.executor = NodeExecutor(cfg, memory=memory, logger=logger, pipeline_runner=runner, tree=self.tree)
        self.scorer = ReasoningNodeScorer(cfg, memory=memory, logger=logger)
        self.mapper = SymbolMapper(cfg, memory=memory, logger=logger)
        self.selector = StepSelector()
        self.builder = FinalPromptBuilder()

    async def run(self, context:dict) -> dict:
        root_id = self.tree.add_root(context.get("goal").get("goal_text"), "Start solving this problem.", "Generate initial plan")
        node = self.tree.nodes[root_id]

        for iteration in range(5):  # Max iterations
            result = await self.executor.execute(node, context)
            node.response = result["response"]
            merged = {
                "thought": node.thought,
                **context
            }
            score = self.scorer.score(node, merged)
            node.score = score.aggregate()
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

        context["final_prompt"] = final_prompt
        return context