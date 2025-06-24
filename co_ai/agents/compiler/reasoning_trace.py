# co_ai/agents/compiler/reasoning_trace.py
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from uuid import uuid4

@dataclass
class ReasoningNode:
    id: str
    parent_id: Optional[str]
    goal: str
    thought: str
    action: str
    response: str
    score: float = 0.0
    children: List["ReasoningNode"] = None
    metadata: Dict = None

class ReasoningTree:
    def __init__(self):
        self.nodes: Dict[str, ReasoningNode] = {}
        self.root_id: Optional[str] = None

    def add_root(self, goal: str, thought: str, action: str) -> str:
        node = ReasoningNode(
            id=str(uuid4()),
            parent_id=None,
            goal=goal,
            thought=thought,
            action=action,
            response="",
            children=[]
        )
        self.nodes[node.id] = node
        self.root_id = node.id
        return node.id

    def add_child(self, parent_id: str, thought: str, action: str, response: str, score: float) -> str:
        node = ReasoningNode(
            id=str(uuid4()),
            parent_id=parent_id,
            goal="",  # Inherited from root
            thought=thought,
            action=action,
            response=response,
            score=score,
            children=[],
            metadata={}
        )
        self.nodes[node.id] = node
        self.nodes[parent_id].children.append(node)
        return node.id

    def get_best_path(self, top_k: int = 3) -> List[ReasoningNode]:
        # DFS or BFS to find highest-scoring paths
        pass