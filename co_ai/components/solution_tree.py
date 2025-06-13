# --- components/solution_tree.py ---

class SolutionNode:
    def __init__(self, plan, code, metric, output, valid):
        self.plan = plan
        self.code = code
        self.metric = metric
        self.output = output
        self.valid = valid

class SolutionTree:
    def __init__(self):
        self.nodes = []

    def initialize(self, goal):
        self.goal = goal
        self.nodes.clear()

    def add_node(self, node):
        self.nodes.append(node)

    def get_best(self):
        valid_nodes = [n for n in self.nodes if n.valid]
        return max(valid_nodes, key=lambda n: n.metric, default=None)

    def get_buggy(self):
        return [n for n in self.nodes if not n.valid]

    def get_valid(self):
        return [n for n in self.nodes if n.valid]


# --- components/search_policy.py ---

import random


class TreeSearchPolicy:
    def __init__(self, config):
        self.n_init = config.get("n_init", 3)
        self.p_debug = config.get("p_debug", 0.2)
        self.p_greedy = config.get("p_greedy", 0.6)

    def select(self, tree):
        draft_count = len(tree.nodes)

        if draft_count < self.n_init:
            return None, "draft"

        if random.random() < self.p_debug:
            buggy = tree.get_buggy()
            if buggy:
                return random.choice(buggy), "debug"

        if random.random() < self.p_greedy:
            valid = tree.get_valid()
            if valid:
                return max(valid, key=lambda n: n.metric), "improve"
        else:
            valid = tree.get_valid()
            if valid:
                return random.choice(valid), "improve"

        return None, "draft"


# --- components/coding_strategy.py ---

class SelfAdaptiveCoder:
    def __init__(self, config):
        self.threshold = config.get("complexity_threshold", 3)

    def score_complexity(self, plan):
        # Placeholder: use LLM or rubric to rate complexity (1–5)
        return 4 if "multi-stage" in plan.lower() else 2

    def generate_code(self, plan):
        complexity = self.score_complexity(plan)
        if complexity <= self.threshold:
            return self._generate_one_pass(plan)
        else:
            return self._generate_stepwise(plan)

    def _generate_one_pass(self, plan):
        return f"# One-pass code for plan\n# {plan}"

    def _generate_stepwise(self, plan):
        steps = [f"Step {i+1}: logic" for i in range(3)]  # Placeholder decomposition
        integrated_code = "\n".join([f"# {s}" for s in steps])
        return f"# Stepwise code for complex plan\n{integrated_code}"
