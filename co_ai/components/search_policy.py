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


