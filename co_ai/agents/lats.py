import math
import re
from collections import defaultdict

import dspy
from dspy import Predict, BootstrapFewShot, Example
from dspy.signatures import InputField, OutputField

from co_ai.agents import BaseAgent
from co_ai.constants import GOAL, PIPELINE_RUN_ID
from co_ai.models import HypothesisORM
from co_ai.agents.mixins.scoring_mixin import ScoringMixin
from co_ai.utils.graph_tools import build_mermaid_graph, compare_graphs, analyze_graph_impact


class TraceStep(dspy.Signature):
    """
    Signature for each reasoning step in LATS.
    """
    state: str = InputField()
    trace: str = InputField()
    next_step: str = OutputField()


class DSPyLATSProgram(dspy.Module):
    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.generator = Predict(TraceStep)
        self.max_depth = cfg.get("max_depth", 3)

    def forward(self, state, trace):
        steps = []
        for step_num in range(self.max_depth):
            self.agent.logger.log("LATSForwardStepStart", {
                "step_num": step_num,
                "current_state_snippet": state[:100],
                "current_trace": trace
            })

            prediction = self.generator(state=state, trace="\n".join(trace))
            if not prediction or not prediction.next_step:
                self.agent.logger.log("LATSNoNextStep", {"step_num": step_num})
                break

            next_step = prediction.next_step.strip()
            trace.append(next_step)
            state = self.agent._update_state(state, next_step)
            steps.append((state, next_step))
            self.agent.logger.log("LATSNextStepGenerated", {
                "step_num": step_num,
                "next_step": next_step,
                "new_state_snippet": state[:100]
            })
            if self.agent.is_terminal({'state': state, 'trace': trace}):
                self.agent.logger.log("LATSTerminalReached", {
                    "step_num": step_num,
                    "final_state_snippet": state[:100]
                })
                break
        return trace, steps


class LATSAgent(ScoringMixin, BaseAgent):
    """
    DSPy-enabled version of LATSAgent with training and comparison.
    """
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        # Setup DSPy
        lm = dspy.LM(
            "ollama_chat/qwen3",
            api_base="http://localhost:11434",
            api_key="",
        )
        dspy.configure(lm=lm)
        self.program = DSPyLATSProgram(cfg, self)

    async def run(self, context: dict) -> dict:
        goal = context[GOAL]
        root_state = goal["goal_text"]
        trace, steps = self.program.forward(state=root_state, trace=[])

        hypothesis = HypothesisORM(
            goal_id=goal["id"],
            source=self.name,
            text="\n".join(trace),
            metadata={"trace": trace, "steps": steps},
            pipeline_run_id=context.get(PIPELINE_RUN_ID),
        )
        self.memory.hypotheses.insert(hypothesis)
        context["lats_result"] = hypothesis.to_dict()
        return context

    def _update_state(self, state, action):
        return state + "\n" + action

    def is_terminal(self, node):
        return "success" in node['state'].lower() or len(node['trace']) >= self.cfg.get("max_depth", 3)

    def train_on_examples(self, examples):
        training_set = [
            Example(state=e["state"], trace="\n".join(e["trace"]), next_step=e["next_step"]).with_inputs("state", "trace")
            for e in examples
        ]
        tuner = BootstrapFewShot(metric=self._trace_quality_metric)
        self.program.generator = tuner.compile(student=Predict(TraceStep), trainset=training_set)

    def _trace_quality_metric(self, example, pred, trace=None):
        if not pred.next_step:
            return 0.0
        # Optionally plug in scoring logic
        return 1.0 if len(pred.next_step.strip()) > 0 else 0.0


class SymbolicImpactAnalyzer:
    """
    Analyzes structural overlap and divergence between two graph representations (e.g., symbolic vs. LATS)
    and attributes score delta to divergent paths.
    """
    def __init__(self, score_lookup_fn):
        self.score_lookup_fn = score_lookup_fn  # Function to get scores for a given node or trace

    def analyze(self, graph1, graph2):
        matches, only_1, only_2 = compare_graphs(graph1, graph2)
        results = []

        for node in matches:
            score_1 = self.score_lookup_fn(node, source="graph1")
            score_2 = self.score_lookup_fn(node, source="graph2")
            delta = score_2 - score_1
            results.append({"node": node, "type": "converged", "delta": delta})

        for node in only_1:
            score = self.score_lookup_fn(node, source="graph1")
            results.append({"node": node, "type": "diverged_graph1", "score": score})

        for node in only_2:
            score = self.score_lookup_fn(node, source="graph2")
            results.append({"node": node, "type": "diverged_graph2", "score": score})

        return results

    def create_node(self, state, trace):
        return {
            "state": state,
            "trace": trace,
            "visits": 0,
            "reward": 0.0,
            "children": [],
            "parent": None,
        }

    def best_uct(self, node):
        def uct(child):
            if child["visits"] == 0:
                return float("inf")
            return child["reward"] / child["visits"] + self.ucb_weight * math.sqrt(
                math.log(node["visits"]) / child["visits"]
            )
        return max(self.children[id(node)], key=uct)