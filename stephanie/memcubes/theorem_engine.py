class TheoremEngine:
    def __init__(self, memory, belief_graph):
        self.memory = memory
        self.belief_graph = belief_graph
        self.validator = TheoremValidator(self.memory.ebt)

    def _find_root(self):
        return next(iter(self.belief_graph.nodes))

    def _find_goal_node(self):
        return next(iter(self.belief_graph.nodes))

    def _build_theorem(self, path):
        # Logic to build a theorem from the path
        pass

    def _is_valid_theorem(self, theorem):
        return self.validator.validate(theorem)

    # In theorem_engine.py
    def extract_theorems(self, belief_graph: nx.DiGraph):
        """Extract validated theorems from belief graph"""
        theorems = []
        for path in nx.all_simple_paths(belief_graph, source=self._find_root(), target=self._find_goal_node()):
            theorem = self._build_theorem(path)
            if self._is_valid_theorem(theorem):
                theorems.append(theorem)
        
        # Save to database
        for theorem in theorems:
            self.memory.db.execute(
                "INSERT INTO theorems VALUES (...)",
                theorem.to_dict()
            )
        return theorems