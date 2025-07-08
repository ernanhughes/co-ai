# stephanie/scoring/scorable_factory.py

from enum import Enum as PyEnum

from stephanie.models.prompt import PromptORM
from stephanie.models.theorem import CartridgeORM
from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.theorem import TheoremORM
from stephanie.models.document import DocumentORM
from stephanie.scoring.scorable import Scorable

# Enum defining all the supported types of scoreable targets
class TargetType(PyEnum):
    DOCUMENT = "document"
    HYPOTHESIS = "hypothesis" 
    CARTRIDGE = "cartridge"
    TRIPLE = "triple"
    CHUNK = "chunk"
    PROMPT = "prompt"
    RESPONSE = "response"
    PROMPT_RESPONSE = "prompt_response"
    TRAINING = "training"
    THEOREM = "theorem"
    SYMBOLIC_RULE = "symbolic_rule"
    CUSTOM = "custom"

class ScorableFactory:
    """
    A factory class that converts various ORM model types into a unified `Scorable` abstraction.
    This allows the scoring system to treat many different content types the same way.
    """

    @staticmethod
    def from_orm(obj, mode: str = "default") -> Scorable:
        """
        Convert an ORM object to a Scorable.
        Dispatches based on the object's class type.
        """
        if isinstance(obj, PromptORM):
            return ScorableFactory.from_prompt_pair(obj, mode)
        elif isinstance(obj, CartridgeORM):
            return Scorable(id=obj.id, text=obj.markdown_content, target_type=TargetType.CARTRIDGE)
        elif isinstance(obj, CartridgeTripleORM):
            # For a triple, we concatenate subject, relation, and object as a textual representation
            return Scorable(id=obj.id, text=f"{obj.subject} {obj.relation} {obj.object}", target_type=TargetType.TRIPLE)
        elif isinstance(obj, TheoremORM):
            return Scorable(id=obj.id, text=obj.statement, target_type=TargetType.THEOREM)
        elif isinstance(obj, DocumentORM):
            # Try summary first, fallback to content or title if missing
            return Scorable(id=obj.id, text=obj.summary or obj.content or obj.title, target_type=TargetType.DOCUMENT)
        else:
            raise ValueError(f"Unsupported ORM type for scoring: {type(obj)}")

    @staticmethod
    def from_prompt_pair(obj: PromptORM, mode: str = "prompt+response") -> Scorable:
        """
        Handles PromptORM objects that contain both prompt and response.
        The `mode` parameter controls whether to extract only the prompt, only the response,
        or a concatenated version of both.
        """
        prompt = obj.prompt or ""
        response = obj.response or ""
        target_type = TargetType.PROMPT

        if mode == "prompt_only":
            text = prompt
        elif mode == "response_only":
            text = response
            target_type = TargetType.RESPONSE
        elif mode == "prompt+response":
            text = f"{prompt}\n\n{response}"
            target_type = TargetType.PROMPT_RESPONSE
        else:
            raise ValueError(f"Invalid prompt scoring mode: {mode}")

        return Scorable(id=obj.id, text=text, target_type=target_type)

    @staticmethod
    def from_dict(data: dict) -> Scorable:
        """
        Creates a Scorable from a raw dictionary. Useful for loading from JSON or manual input.
        Example input:
            {
                "id": 123,
                "text": "This is a hypothesis about climate change.",
                "target_type": "hypothesis"
            }
        Tries to map the string 'target_type' to a known TargetType, otherwise defaults to CUSTOM.
        """
        target_type_str = data.get("target_type", "Custom")

        try:
            target_type = TargetType(target_type_str)
        except ValueError:
            target_type = TargetType.CUSTOM

        return Scorable(
            id=data.get("id"),
            text=data.get("text", ""),
            target_type=target_type
        )
