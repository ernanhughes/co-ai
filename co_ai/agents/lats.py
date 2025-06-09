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
            reward = self.simulate(node)
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

    def expand(self, node, context:dict):
        # Build prompt using current node's state and trace
        merged = {
            **context,
            "state": node["state"],
            "trace": node["trace"]
        }
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)  # ← Load prompt template

        # Call LLM to generate n completions
        response = self.call_llm(prompt, context=merged)  # ← Call LLM here

        # Parse responses into multiple thoughts/actions
        completions = self._parse_completions(response)

        # Create children nodes
        children = []
        for comp in completions:
            new_state = self._update_state(node['state'], comp)
            new_trace = node['trace'] + [comp]
            child = self.create_node(new_state, new_trace, parent=node)
            children.append(child)

        self.children[id(node)] = children
        return children[0]  # Return one of the children

    def simulate(self, node):
        # Simulate until terminal or max depth
        current = node
        while not self.is_terminal(current) and len(current['trace']) < self.max_depth:
            prompt = self._build_prompt(current)
            response = self.call_llm(prompt, {})
            action = self._choose_action(response)
            new_state = self._update_state(current['state'], action)
            current = self.create_node(new_state, current['trace'] + [action], parent=current)

        # Get external feedback or reward
        reward = self.evaluate(current)
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