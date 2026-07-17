from typing import List
from llm_sdk import Small_LLM_Model


class LLM:
    llm: Small_LLM_Model

    def __init__(self) -> None:
        self.llm = Small_LLM_Model()

    def encode(self, prompt: str) -> List[int]:
        # print(f"prompt is \n--------\n{prompt}\n--------")
        return (self.llm.encode(prompt)).tolist()[0]

    def decode(self, id: int) -> str:
        return self.llm.decode([id])

    def get_logits(self, input_ids: List[int]) -> List[float]:
        return self.llm.get_logits_from_input_ids(input_ids)
