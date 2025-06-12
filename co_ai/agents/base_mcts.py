from abc import ABC, abstractmethod
import math
import uuid

class LATSNode:
    """Unified node structure for MCTS"""
    def __init__(self, state, trace, parent=None):
        self.id = str(uuid.uuid4())
        self.state = state
        self.trace = trace
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.score = None
        self.dimension_scores = {}
        self.is_terminal = False

    def is_leaf(self):
        return len(self.children) == 0


class BaseMCTSAgent(ABC):
    """Abstract base class for MCTS-based agents"""
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = None
        self.nodes = []
        self.max_depth = cfg.get("max_depth", 5)
        self.branching_factor = cfg.get("branching_factor", 3)
        self.ucb_weight = cfg.get("ucb_weight", 1.41)
        self.num_simulations = cfg.get("num_simulations", 50)
        self.prune_threshold = cfg.get("prune_threshold", 0.4)

    def create_node(self, state, trace, parent=None):
        """Create a new node with proper structure"""
        node = LATSNode(state, trace, parent)
        self.nodes.append(node)
        return node

    def _is_terminal(self, node: LATSNode) -> bool:
        """Check if node is terminal state"""
        return node.is_terminal or len(node.trace) >= self.max_depth

    def _backpropagate(self, node: LATSNode, reward: float):
        """Update node statistics up the tree"""
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def _uct_score(self, parent_visits: int, child: LATSNode):
        """Calculate UCT score for node selection"""
        if child.visits == 0:
            return float('inf')
        return (child.reward / child.visits) + \
               self.ucb_weight * math.sqrt(math.log(parent_visits) / child.visits)

    def _best_uct(self, node: LATSNode):
        """Select best child using UCT formula"""
        return max(node.children, key=lambda c: self._uct_score(node.visits, c))

    def _select(self, node: LATSNode):
        """Select node for expansion using UCT"""
        while not self._is_terminal(node) and not node.is_leaf():
            node = self._best_uct(node)
        return node

    def _expand(self, node: LATSNode, context: dict):
        """Generate children nodes using agent-specific expansion"""
        completions = self._generate_completions(node, context)
        
        for comp in completions:
            new_state = self._update_state(node.state, comp)
            new_trace = node.trace + [comp]
            

                        # Add mode to context
            score_context = {
                "goal": {"goal_text": context["goal"]["goal_text"]}
            }

            # Generate dimension scores
            score_result = self._score_hypothesis(
                {"text": comp},
                score_context,
                metrics="lats_node"
            )
            
            child = self.create_node(new_state, new_trace, parent=node)
            child.score = score_result.get("score", 0.0)
            child.dimension_scores = score_result.get("scores", {})
            child.reward = child.score  # Initialize reward
            node.children.append(child)

    @abstractmethod
    def _generate_completions(self, node: LATSNode, context: dict):
        """Agent-specific expansion logic"""
        pass

    @abstractmethod
    def _score_hypothesis(self, hypothesis: dict, context: dict, metrics: str = "lats_node"):
        """Agent-specific scoring logic"""
        pass

    @abstractmethod
    def _update_state(self, state, action: str):
        """Agent-specific state update logic"""
        pass

    def run(self, context: dict):
        """Main MCTS loop"""
        # Initialize root node
        goal = context["goal"]
        self.root = self.create_node(goal["goal_text"], [])
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = self._select(self.root)
            if not self._is_terminal(node):
                self._expand(node, context)
            reward = self._simulate(node, context)
            self._backpropagate(node, reward)
        
        # Return best path
        best_child = self._best_uct(self.root)
        return {
            "trace": best_child.trace,
            "score": best_child.score,
            "dimension_scores": best_child.dimension_scores
        }