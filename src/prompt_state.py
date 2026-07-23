from typing import List
from .input_reader import Function


class PromptState:
    logits: List[float]
    generated_tokens: str
    functions: List[Function]

    def __init__(self, functions: List[Function]):
        self.logits = []
        self.generated_tokens = ""
        self.functions = functions
