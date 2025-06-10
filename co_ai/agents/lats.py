import math
import re
from collections import defaultdict

from co_ai.agents import BaseAgent
from co_ai.constants import GOAL, PIPELINE_RUN_ID
from co_ai.models import HypothesisORM
from co_ai.agents.mixins.scoring_mixin import ScoringMixin


class LATSAgent(ScoringMixin, BaseAgent):
    """
    Implements the core logic of the LATS (Language Agent Tree Search) paper:
    Iteratively generates actions, evaluates outcomes, and plans next steps
    based on a tree-like search pattern over reasoning traces.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.max_depth = cfg.get("max_depth", 3)
        self.branching_factor = cfg.get("branching_factor", 2)
        self.ucb_weight = cfg.get("ucb_weight", 1.41)
        self.num_simulations = cfg.get("num_simulations", 50)

        # Node tracking
        self.N = defaultdict(int)  # visit count
        self.W = defaultdict(float)  # total reward
        self.children = dict()  # node -> children

    async def run(self, context: dict) -> dict:
        goal = context[GOAL]
        root_state = goal["goal_text"]
        root = self.create_node(state=root_state, trace=[])
        for _ in range(self.num_simulations):
            node = self.select(root)
            if not self.is_terminal(node):
                node = self.expand(node, context)
            reward = self.simulate(node, context)
            self.backpropagate(node, reward)

        best_child = self.best_uct(node=root, ucb_weight=0)  # greedy
        best_trace = best_child['trace']
        
        hypothesis = HypothesisORM(
            goal_id=goal["id"],
            source=self.name,
            text="\n".join(best_trace),
            metadata={"trace": best_trace},
            pipeline_run_id=context.get(PIPELINE_RUN_ID),
        )
        self.memory.hypotheses.insert(hypothesis)
        context["lats_result"] = hypothesis.to_dict()
        return context


    def select(self, node):
        while self.children.get(id(node)):
            unvisited = [c for c in self.children[id(node)] if node['visits'] == 0]
            if unvisited:
                return unvisited[0]
            node = self.best_uct(node)
        return node

    def best_uct(self, node, ucb_weight=1.41):
        def uct(child):
            if child['visits'] == 0:
                return float('inf')
            return (child['reward'] / child['visits']) + \
                ucb_weight * math.sqrt(math.log(node['visits']) / child['visits'])
        return max(self.children[id(node)], key=uct)

    def expand(self, node, context: dict):
        merged = {
            **context,
            "state": node["state"],
            "trace": node["trace"],
        }
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)
        print(f"formatted_prompt: {prompt}")
        # Call LLM to get completions
        response = self.call_llm(prompt, context=merged)
        print(f"LLM Response: {response}")
        completions = self._parse_completions(response)

        children = []
        for comp in completions:
            new_state = self._update_state(node['state'], comp)
            new_trace = node['trace'] + [comp]

            # Create hypothesis-like object
            hyp = {
                "text": comp,
                "id": f"hyp_{len(self.children)}",
                "goal_id": context[GOAL]["id"]
            }

            # Score using custom dimension set
            score_result = self.score_hypothesis(hyp, context, metrics="lats_node")

            # Build child node with score metadata
            child = self.create_node(new_state, new_trace, parent=node)
            child["score"] = score_result["score"]
            child["dimension_scores"] = score_result["scores"]

            children.append(child)

        self.children[id(node)] = children
        return children[0]
    
    def simulate(self, node, context):
        current = node
        while not self.is_terminal(current) and len(current['trace']) < self.max_depth:
            prompt = self._build_prompt(current)
            response = self.call_llm(prompt, {})
            action = self._choose_action(response)
            new_state = self._update_state(current['state'], action)
            current = self.create_node(new_state, current['trace'] + [action], parent=current)

        # Evaluate final node using reflection scorer
        hyp = {
            "text": "\n".join(current['trace']),
            "id": f"hyp_final_{id(current)}",
            "goal_id": current['state'].get("goal_id")
        }

        score_result = self.score_hypothesis(hyp, context, metrics="lats_reflection")
        reward = score_result["score"] / 100  # Normalize to 0–1 range
        return reward

    def evaluate(self, node):
        # Can be replaced with environment call
        # For now, simulate success/failure
        return 1.0 if "success" in node['state'].lower() else 0.0


    def backpropagate(self, node, reward):
        while node:
            node['visits'] += 1
            node['reward'] += reward
            node = node['parent']

    def is_terminal(self, node):
        return "success" in node['state'].lower() or len(node['trace']) >= self.max_depth

    def _build_prompt(self, node):
        merged = {"state": node['state'], "trace": node['trace']}
        return self.prompt_loader.load_prompt(self.cfg, merged)

    def _choose_action(self, response):
        completions = self._parse_completions(response)
        return completions[0] if completions else ""

    def create_node(self, state, trace, parent=None):
        return {
            'state': state,
            'trace': trace,
            'parent': parent,
            'visits': 0,
            'reward': 0.0,
            'children': []
        }
    
    def _parse_completions(self, response: str) -> list:
        """
        Parses an LLM response containing multiple thoughts/actions into a list.
        
        Supports formats like:
            "Thought 1: ...\nThought 2: ..."
            "- ...\n- ..."
            "• ...\n• ..."

        Returns:
            List[str]: Parsed completions, limited by branching factor.
        """
        # Match patterns like "Thought 1:", "Thought 2:", etc.
        thought_pattern = r"([Tt]hought\s*\d+|[Aa]ction\s*\d+|[-•])\s*(.*?)(?=\n(?:[Tt]hought\s*\d+|[Aa]ction\s*\d+|[-•])\s|\Z)"
        matches = re.findall(thought_pattern, response.strip(), re.DOTALL)

        # If no structured format found, split by line
        if not matches:
            lines = [line.strip() for line in response.strip().split('\n')]
            return [line for line in lines if line][:self.branching_factor]

        # Extract just the content part from each match
        completions = [match[-1].strip() for match in matches if match[-1].strip()]

        return completions[:self.branching_factor]

    def _update_state(self, state: dict, action: str) -> dict:
        observation = self.env.step(action)  # Get feedback from environment
        new_state = state.copy()
        new_state["history"].append({
            "action": action,
            "observation": observation
        })
        return new_state