import json
from typing import List
from .json_extractors import LiteralJSONExtractor
from .llm import LLM
from .models import Function, Prompt


class FunctionNameExtractor:
    """Extracts function names from prompts using constrained JSON generation.

    Attributes:
        llm (LLM): Model instance used for guided output generation.
    """

    llm: LLM

    def __init__(self, llm: LLM) -> None:
        """Initializes the extractor with an LLM instance.

        Args:
            llm (LLM): Language model wrapper.
        """
        self.llm = llm

    def extract(
        self, prompt: Prompt, functions: List[Function]
    ) -> str:
        """Identifies the matching function name for a prompt.

        Args:
            prompt (Prompt): Container holding user prompt text.
            functions (List[Function]): List of candidate functions.

        Returns:
            str: JSON-encoded string of the matched function name.
        """
        base_prompt = FunctionNameExtractor.make_function_prompt(
            prompt.prompt, functions
        )
        allowed_functions = FunctionNameExtractor.get_json_function_names(
            functions
        )

        name: str = LiteralJSONExtractor(self.llm, allowed_functions).extract(
            base_prompt, prompt.prompt
        )

        return name

    @staticmethod
    def make_function_prompt(prompt: str, functions: List[Function]) -> str:
        """Constructs the full extraction prompt with function definitions.

        Args:
            prompt (str): Raw user prompt text.
            functions (List[Function]): List of supported functions.

        Returns:
            str: Formatted system prompt with examples and schema.
        """
        return (
            "For the given prompt, reply in JSON string with the function " +
            "name.\n\n" +
            "Examples:\n" +
            'What is the sum of 100 and 50? → "fn_add_numbers"\n' +
            'Greet gandalf → "fn_greet"\n' +
            'Replace all vowels in "hello" with asterisks → ' +
            '"fn_substitute_string_with_regex"\n' +
            f"\n\nThe following functions are supported: {json.dumps([
                {"name": f.name, "description": f.description}
                for f in functions
            ])}" +
            f'\n\nPrompt: {prompt}\nOutput: '
        )

    @staticmethod
    def get_json_function_names(functions: List[Function]) -> List[str]:
        """Serializes function names into JSON-compatible strings.

        Args:
            functions (List[Function]): Supported function models.

        Returns:
            List[str]: JSON-encoded function names.
        """
        return [json.dumps(f.name) for f in functions]
