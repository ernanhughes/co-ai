# stephanie/scoring/scorable_factory.py

from enum import Enum as PyEnum

from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.document import DocumentORM
from stephanie.models.prompt import PromptORM
from stephanie.models.theorem import CartridgeORM, TheoremORM
from stephanie.scoring.scorable import Scorable


class ScorableFactory:
    @staticmethod 
    def from_orm(obj) -> Scorable: 
        if isinstance(obj, PromptORM):
            return Scorable(
                id=obj.id,
                text=obj.prompt + "\n\n" + obj.response,
                target_type="Prompt"
            )
        elif isinstance(obj, CartridgeORM):
            return Scorable(
                id=obj.id,
                text=obj.markdown_content,
                target_type="Cartridge"
            )
        elif isinstance(obj, CartridgeTripleORM):
            return Scorable(
                id=obj.id,
                text=f"{obj.subject} {obj.relation} {obj.object}",
                target_type="Triple"
            )
        elif isinstance(obj, TheoremORM):
            return Scorable(
                id=obj.id,
                text=obj.statement,
                target_type="Theorem"
            )
        elif isinstance(obj, DocumentORM):
            return Scorable(
                id=obj.id,
                text=obj.summary or obj.content or obj.title,
                target_type="Document"
            )
        else:
            raise ValueError(f"Unsupported ORM type for scoring: {type(obj)}")


class TargetType(PyEnum):
    DOCUMENT = "document"
    HYPOTHESIS = "hypothesis"
    CARTRIDGE = "cartridge"
    TRIPLE = "triple"
    CHUNK = "chunk"
    PROMPT = "prompt"
    RESPONSE = "response"
    TRAINING = "training"
    THEOREM = "theorem"
    SYMBOLIC_RULE = "symbolic_rule"
