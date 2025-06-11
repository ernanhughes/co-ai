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
from co_ai.agents.proximity import ProximityAgent
from co_ai.utils.graph_tools import (
    build_mermaid_graph,
    compare_graphs,
    analyze_graph_impact,
)
import math
from collections import defaultdict
import json
import re

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, PIPELINE_RUN_ID
from co_ai.models import HypothesisORM, EvaluationORM
from co_ai.agents.mixins.scoring_mixin import ScoringMixin
from co_ai.agents.proximity import ProximityAgent
from co_ai.utils.graph_tools import build_mermaid_graph, compare_graphs
from co_ai.agents.unified_mrq import UnifiedMRQAgent
from co_ai.agents.rule_tuner import RuleTunerAgent


from dspy import Signature, InputField, OutputField


class TraceStep(Signature):
    """
    A reasoning step in the LATS framework.

    Inputs:
        - state: Current problem state (e.g., goal + history)
        - trace: Sequence of previous thoughts/actions

    Outputs:
        - next_step: Next thought/action to explore
    """

    state = InputField(desc="Current problem state")
    trace = InputField(desc="History of thoughts/actions taken so far")
    next_step = OutputField(desc="Next reasoning step (thought or action)")


class ReflectionPrompt(Signature):
    """
    Self-reflection module to analyze failed reasoning paths.

    Inputs:
        - state: Final state after failed attempt
        - trace: Full reasoning path
        - goal: Original goal text

    Outputs:
        - rationale: Explanation of failure
        - improvement_plan: Suggested improvements
    """

    state = InputField(desc="Final state after failed attempt")
    trace = InputField(desc="Full reasoning path")
    goal = InputField(desc="Original goal text")

    rationale = OutputField(desc="Why the attempt failed")
    improvement_plan = OutputField(desc="Concrete steps to improve")


class ValueEstimator(Signature):
    """
    Evaluates a reasoning path using a hybrid value function.

    Inputs:
        - state: Current problem state
        - trace: Reasoning steps taken
        - goal: Goal text

    Outputs:
        - score: Normalized score (0–1)
        - rationale: Explanation of the score
    """

    state = InputField(desc="Current problem state")
    trace = InputField(desc="Sequence of thoughts/actions")
    goal = InputField(desc="Goal text")

    score = OutputField(desc="Hybrid score (LM + self-consistency)")
    rationale = OutputField(desc="Explanation of score")


class SharpeningPrompt(Signature):
    """
    Sharpens hypotheses using dimensional feedback.

    Inputs:
        - hypothesis: Original hypothesis text
        - feedback: Dimensional scores and rationales
        - goal: Original goal

    Outputs:
        - refined_hypothesis: Improved version
        - changes: Summary of changes made
    """

    hypothesis = InputField(desc="Original hypothesis")
    feedback = InputField(desc="Dimensional scores and rationales")
    goal = InputField(desc="Goal text")

    refined_hypothesis = OutputField(desc="Improved hypothesis")
    changes = OutputField(desc="Summary of changes made")


class LATSProgram(dspy.Module):
    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.generator = Predict(TraceStep)
        self.value_estimator = Predict(ValueEstimator)
        self.reflector = Predict(ReflectionPrompt)
        self.sharpener = Predict(SharpeningPrompt)
        self.max_depth = cfg.get("max_depth", 3)

    def _estimate_value(self, state, trace):
        """Estimate value using LM-powered scorer"""
        result = self.value_estimator(state=state, trace=trace, goal=state)
        try:
            score = float(result.score)
        except:
            score = 0.5
        return score, result.rationale

    def forward(self, state, trace, depth=0):
        if depth >= self.max_depth:
            return trace, self._estimate_value(state, trace)[0]

        prediction = self.generator(state=state, trace=trace)
        if not prediction or not prediction.next_step:
            return trace, 0.0

        next_step = prediction.next_step.strip()
        new_state = self.agent._update_state(state, next_step)
        new_trace = trace + [next_step]

        child_trace, child_score = self.forward(new_state, new_trace, depth + 1)

        if child_score < self.cfg.get("threshold", 0.7):
            reflection = self.reflector(state=new_state, trace=child_trace, goal=state)
            sharpened = self.sharpener(
                hypothesis=next_step, feedback=reflection.rationale, goal=state
            )
            child_trace[-1] = sharpened.refined_hypothesis
            new_state = self.agent._update_state(state, child_trace[-1])
            score, _ = self._estimate_value(new_state, child_trace)
            return child_trace, score

        return child_trace, child_score


class LATSAgent(ScoringMixin, BaseAgent):
    """
    Enhanced LATS agent with:
    - Tree search (MCTS + UCT)
    - Multi-dimensional scoring
    - Proximity-based reuse
    - Reflection/refinement
    - Rule tuning
    - DSPy optimization
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.max_depth = cfg.get("max_depth", 5)
        self.branching_factor = cfg.get("branching_factor", 3)
        self.ucb_weight = cfg.get("ucb_weight", 1.41)
        self.num_simulations = cfg.get("num_simulations", 50)
        self.lambda_weight = cfg.get("lambda", 0.5)

        # Node tracking
        self.nodes = []
        self.N = defaultdict(int)  # visit count
        self.W = defaultdict(float)  # total reward
        self.children = dict()  # node -> children

        # Initialize sub-agents
        self.proximity_agent = ProximityAgent(
            cfg.get("proximity", {}), memory=memory, logger=logger
        )
        self.rule_tuner = RuleTunerAgent(
            cfg.get("rule_tuner", {}), memory=memory, logger=logger
        )
        self.mrq_agent = UnifiedMRQAgent(
            cfg.get("mrq", {}), memory=memory, logger=logger
        )

        # Setup DSPy
        lm = dspy.LM(
            "ollama_chat/qwen3",
            api_base="http://localhost:11434",
            api_key="",
        )
        dspy.configure(lm=lm)

        # Initialize DSPy program
        self.lats_program = LATSProgram(cfg, self)

        # Symbolic impact analyzer
        self.impact_analyzer = SymbolicImpactAnalyzer(self._get_score)

    async def run(self, context: dict) -> dict:
        """Main LATS search loop"""
        goal = context[GOAL]
        root_state = goal["goal_text"]

        # 1. Initialize root node
        root = self.create_node(state=root_state, trace=[], parent=None)

        # 2. Run MCTS simulations
        for sim_num in range(self.num_simulations):
            # Selection
            node = self.select(root)

            # Expansion
            if not self.is_terminal(node):
                node = await self.expand(node, context)

            # Simulation & Evaluation
            reward, trace_data = self.simulate_and_evaluate(node, context)

            # Backpropagation
            self.backpropagate(node, reward, trace_data)

            # Optional: Periodic refinement
            if sim_num % 10 == 0:
                await self._refine_system(context)

        # 3. Get best path
        best_child = self.best_uct(node=root, ucb_weight=0)  # Greedy selection
        best_trace = best_child["trace"]

        # 4. Create final hypothesis
        hypothesis = HypothesisORM(
            goal_id=goal["id"],
            source=self.name,
            text="\n".join(best_trace),
            metadata={
                "trace": best_trace,
                "path": [n["id"] for n in best_trace],
                "scores": {
                    dim: best_child["dimension_scores"][dim]["score"]
                    for dim in best_child["dimension_scores"]
                },
            },
            pipeline_run_id=context.get(PIPELINE_RUN_ID),
        )

        self.memory.hypotheses.insert(hypothesis)
        context["lats_result"] = hypothesis.to_dict()
        return context

    def create_node(self, state, trace, parent=None):
        """Create a new node in the search tree"""
        node = {
            "id": len(self.nodes) + 1,
            "state": state,
            "trace": trace,
            "parent": parent,
            "visits": 0,
            "reward": 0.0,
            "children": [],
            "is_terminal": False,
            "dimension_scores": {},
            "final_score": 0.0,
        }
        self.nodes.append(node)
        return node

    def select(self, node):
        """Select node for expansion using UCT"""
        while self.children.get(id(node)) and self.children[id(node)]:
            unvisited = [c for c in self.children[id(node)] if c["visits"] == 0]
            if unvisited:
                return unvisited[0]
            node = self.best_uct(node)
        return node

    def best_uct(self, node, ucb_weight=None):
        """Select best child using UCT formula"""
        ucb_weight = ucb_weight or self.ucb_weight

        def uct(child):
            if child["visits"] == 0:
                return float("inf")
            return (child["reward"] / child["visits"]) + ucb_weight * math.sqrt(
                math.log(node["visits"]) / child["visits"]
            )

        return max(self.children[id(node)], key=uct)

    async def expand(self, node, context: dict):
        """Generate new children nodes from current node"""
        # Build prompt with context
        merged = {
            **context,
            "state": node["state"],
            "trace": node["trace"],
            "mode": "reason",
        }

        # 1. Get similar hypotheses
        proximity_context = await self._run_proximity(context)
        merged["similar_hypotheses"] = proximity_context.get("most_similar", "")

        # 2. Generate completions with DSPy
        completions, steps = self.lats_program.forward(
            state=node["state"], trace=node["trace"], depth=0
        )

        # 3. Apply proximity-based refinement
        refined_completions = []
        for comp in completions:
            refined = self._apply_proximity_guidance(comp, proximity_context)
            refined_completions.append(refined)

        # 4. Score and build children
        children = []
        for comp in refined_completions:
            new_state = self._update_state(node["state"], comp)
            new_trace = node["trace"] + [comp]

            # Score using dimensional scorers
            hyp = {"text": comp, "goal_id": context[GOAL]["id"]}

            score_result = self.score_hypothesis(hyp, context, metrics="lats_node")

            # Create child node with metadata
            child = self.create_node(state=new_state, trace=new_trace, parent=node)
            child["score"] = score_result["score"]
            child["dimension_scores"] = score_result["scores"]
            child["action"] = comp

            children.append(child)

        # Store children
        self.children[id(node)] = children
        return children[0] if children else node

    def simulate_and_evaluate(self, node, context):
        """Simulate until terminal state and return final reward"""
        current = node
        while not self.is_terminal(current) and len(current["trace"]) < self.max_depth:
            # Build prompt
            merged = {
                **context,
                "state": current["state"],
                "trace": current["trace"],
                "mode": "simulate",
            }
            prompt = self.prompt_loader.load_prompt(self.cfg, merged)
            response = self.call_llm(prompt, context=merged)

            # Parse completions
            completions = self._parse_completions(response)
            if not completions:
                break

            action = completions[0]  # Take first completion
            new_state = self._update_state(current["state"], action)
            new_trace = current["trace"] + [action]

            # Create new node
            current = self.create_node(state=new_state, trace=new_trace, parent=current)

        # Evaluate final node
        reward, trace_data = self.evaluate(current)
        return reward, trace_data

    def evaluate(self, node):
        """Evaluate node using hybrid LM + self-consistency scoring"""
        if self.cfg.get("use_environment", False):
            obs = self.env.step(node["state"])
            return obs["reward"], {"trace": node["trace"], "environment": obs}

        # Fallback: dimensional scoring
        print(node)
        hyp = {
            "text": "\n".join(node["trace"]),
            "goal_id": node["state"].get("goal_id"),
        }

        score_result = self.score_hypothesis(hyp, {}, metrics="lats_reflection")
        return score_result["score"] / 100, score_result

    def backpropagate(self, node, reward, trace_data=None):
        """Update node statistics up the tree"""
        while node:
            node["visits"] += 1
            node["reward"] += reward

            # Store trace data for analysis
            if trace_data:
                node.setdefault("history", []).append(
                    {
                        "visits": node["visits"],
                        "reward": reward,
                        "trace_data": trace_data,
                    }
                )

            node = node["parent"]

    def is_terminal(self, node):
        """Check if node is terminal state"""
        return (
            "success" in node["state"].lower() or len(node["trace"]) >= self.max_depth
        )

    def _update_state(self, state: str, action: str) -> str:
        """Update state with action result"""
        return f"{state}\n{action}"

    def _parse_completions(self, response: str) -> list:
        """Parse multiple thoughts/actions from response"""
        thought_pattern = r"([Tt]hought\s*\d+|[Aa]ction\s*\d+|[-•])\s*(.*?)(?=\n(?:[Tt]hought\s*\d+|[Aa]ction\s*\d+|[-•])\s|\Z)"
        matches = re.findall(thought_pattern, response.strip(), re.DOTALL)

        if not matches:
            return [response.strip()]

        completions = [match[-1].strip() for match in matches if match[-1].strip()]
        return completions[: self.branching_factor]

    async def _run_proximity(self, context):
        """Run proximity agent to find similar hypotheses"""
        try:
            return await self.proximity_agent.run(context)
        except Exception as e:
            self.logger.log("ProximityAgentFailed", {"error": str(e)})
            return {}

    def _apply_proximity_guidance(self, comp, proximity_data):
        """Enhance completion using proximity feedback"""
        if not proximity_data.get("most_similar"):
            return comp

        # Use LLM to refine action with proximity info
        prompt = self.prompt_loader.load_prompt(
            "proximity_guidance",
            {
                "current_action": comp,
                "similar_hypotheses": proximity_data["most_similar"],
            },
        )

        response = self.call_llm(prompt, {})
        return response.strip()

    def _apply_reflection_to_prompt(self, prompt, reflection):
        """Inject reflection into prompt for future steps"""
        if not reflection:
            return prompt

        reflection_prompt = self.prompt_loader.load_prompt(
            "reflection_injection", {"prompt": prompt, "reflection": reflection}
        )

        return self.call_llm(reflection_prompt, {})

    def _get_score(self, node, source="graph1"):
        """Get score for symbolic impact analysis"""
        hyp = {"text": "\n".join(node["trace"]), "id": f"hyp_{node['id']}"}

        score_result = self.score_hypothesis(hyp, {}, metrics="lats_reflection")
        return score_result["score"] / 100  # Normalize

    async def _refine_system(self, context):
        """Refine rules and models using collected data"""
        # 1. Analyze graph impact
        if len(self.nodes) > 1:
            analysis = self.impact_analyzer.analyze(self.nodes[0], self.nodes[-1])
            context["graph_analysis"] = analysis

        # 2. Train MR.Q on high-quality traces
        high_scoring = [n for n in self.nodes if n["score"] > 0.8]
        if high_scoring:
            await self.mrq_agent.run({"traces": high_scoring})

        # 3. Tune symbolic rules based on analysis
        if context.get("graph_analysis"):
            await self.rule_tuner.run(context)

        return context

    def _get_value(self, node):
        """Calculate value using hybrid LM + self-consistency"""
        lm_score = node.get("score", 0.5)
        sc_score = self._self_consistency(node)
        return self.lambda_weight * lm_score + (1 - self.lambda_weight) * sc_score

    def _self_consistency(self, node):
        """Calculate self-consistency score for node"""
        if not node["trace"]:
            return 0.0

        # Use LLM to evaluate consistency
        prompt = self.prompt_loader.load_prompt(
            "self_consistency", {"trace": node["trace"], "state": node["state"]}
        )
        response = self.call_llm(prompt, {})

        # Parse numerical score
        score_match = re.search(r"(\d+)", response)
        return int(score_match.group(1)) / 100 if score_match else 0.5

    def _get_dimension_score(self, trace):
        """Get dimensional scores for trace"""
        # Build hypothesis
        hyp = {"text": "\n".join(trace), "id": f"hyp_{len(self.nodes)}"}

        # Score across dimensions
        score_result = self.score_hypothesis(hyp, {}, metrics="lats_reflection")
        return score_result["score"] / 100  # Normalize

    def _train_on_traces(self, traces):
        """Train DSPy module on high-quality traces"""
        # Convert traces to examples
        examples = [
            Example(
                state=trace["state"],
                trace=trace["trace"],
                next_step=trace["last_action"],
            )
            for trace in traces
        ]

        # Use dimensional scores as weights
        weighted_examples = [
            example.with_score(self._get_dimension_score(example.trace))
            for example in examples
        ]

        # Compile with BootstrapFewShot
        tuner = BootstrapFewShot(metric=self._dimension_aware_metric)
        self.lats_program.generator = tuner.compile(
            student=Predict(TraceStep), trainset=weighted_examples
        )

    def _dimension_aware_metric(self, example, pred):
        """Use dimensional scores for training metric"""
        scores = self._get_dimension_scores(pred.trace)
        return sum(s["score"] * s.get("weight", 1.0) for s in scores.values())

    def _get_dimension_scores(self, trace):
        """Get scores across all dimensions"""
        hyp = {"text": "\n".join(trace), "id": f"hyp_{len(self.nodes)}"}
        return self.score_hypothesis(hyp, {}, metrics="lats_node")

    def _generate_reflection(self, node):
        """Generate reflection for failed trajectory"""
        prompt = self.prompt_loader.load_prompt(
            "reflection",
            {
                "trace": node["trace"],
                "state": node["state"],
                "goal": node["state"],  # Use state as goal proxy
            },
        )
        response = self.call_llm(prompt, {})
        return response.strip()

    def _build_prompt(self, node):
        """Build prompt for node evaluation"""
        merged = {"state": node["state"], "trace": node["trace"], "mode": "evaluate"}
        return self.prompt_loader.load_prompt(self.cfg, merged)

    def _choose_action(self, response):
        """Choose best action from response"""
        completions = self._parse_completions(response)
        return completions[0] if completions else ""

    def _self_consistency_check(self, node):
        """Validate consistency of reasoning path"""
        prompt = self.prompt_loader.load_prompt(
            "self_consistency", {"trace": node["trace"], "state": node["state"]}
        )
        response = self.call_llm(prompt, {})

        # Parse consistency score
        score_match = re.search(r"(\d+)", response)
        return int(score_match.group(1)) / 100 if score_match else 0.5

    def _should_prune(self, node):
        """Determine if node should be pruned"""
        return node.get("score", 0) < self.cfg.get("prune_threshold", 0.4)

    def _get_node_path(self, node):
        """Get full path from root to node"""
        path = []
        while node:
            path.append(node)
            node = node["parent"]
        return path[::-1]  # Reverse to get root-first

    def _log_simulation(self, sim_num, node, reward):
        """Log simulation results for analysis"""
        self.logger.log(
            "LATSIteration",
            {
                "simulation": sim_num,
                "node_id": node["id"],
                "reward": reward,
                "trace": node["trace"][-3:] if node["trace"] else [],
                "depth": len(node["trace"]),
            },
        )


class SymbolicImpactAnalyzer:
    """
    Analyzes structural overlap and divergence between two graph representations (e.g., symbolic vs. LATS)
    and attributes score delta to divergent paths.
    """

    def __init__(self, score_lookup_fn):
        self.score_lookup_fn = (
            score_lookup_fn  # Function to get scores for a given node or trace
        )

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
