from typing import List
from llm_sdk import Small_LLM_Model


class LLM:
    """Wrapper class for interacting with a language model interface.

    Attributes:
        llm (Small_LLM_Model): Instance of the underlying language model.
    """

    llm: Small_LLM_Model

    def __init__(self) -> None:
        """Initializes the LLM instance and loads the underlying model."""
        self.llm = Small_LLM_Model()

    def encode(self, prompt: str) -> List[int]:
        """Encodes a text prompt into a list of token IDs.

        Args:
            prompt (str): Text prompt to encode.

        Returns:
            List[int]: List of generated token IDs.
        """
        return (self.llm.encode(prompt)).tolist()[0]

    def decode(self, id: int) -> str:
        """Decodes a single token ID back into its string representation.

        Args:
            id (int): Token ID to decode.

        Returns:
            str: Decoded text string.
        """
        return self.llm.decode([id])

    def get_logits(self, input_ids: List[int]) -> List[float]:
        """Computes output logits for a sequence of token IDs.

        Args:
            input_ids (List[int]): List of token IDs.

        Returns:
            List[float]: Output logits produced by the model.
        """
        return self.llm.get_logits_from_input_ids(input_ids)
