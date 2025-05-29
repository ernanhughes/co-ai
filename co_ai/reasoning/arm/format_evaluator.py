from co_ai.reasoning.arm import ARMReasoningSelfEvaluator
from typing import Callable

class ARMFormatEvaluator(ARMReasoningSelfEvaluator):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)

    def _generate_with_format(self, prompt: str, fmt: str, context:dict, llm:Callable) -> str:
        """
        Actually generate response using the specified format.
        Wraps the prompt with format tokens before calling the model.
        """
        tag_map = {
            "direct": "Direct",
            "short_cot": "Short_CoT",
            "code": "Code",
            "long_cot": "Long_CoT",
        }

        tag = tag_map.get(fmt, "Direct")
        full_prompt = f"<{tag}>{prompt}</{tag}>"
        response = llm(full_prompt, context)
        self.logger.log(
            "ARMFormatEvaluatorResponse",
            {"prompt": full_prompt, "response": response, "format": fmt}
        )
        return response