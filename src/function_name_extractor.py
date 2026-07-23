import json
from typing import List
from .json_extractors import LiteralJSONExtractor
from .llm import LLM
from .models import Function, Prompt


class FunctionNameExtractor:
    llm: LLM

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def extract(
        self, prompt: Prompt, functions: List[Function]
    ) -> str:
        base_prompt = FunctionNameExtractor.make_function_prompt(
            prompt.prompt, functions
        )
        allowed_functions = FunctionNameExtractor.get_json_function_names(
            functions
        )

        return LiteralJSONExtractor(self.llm, allowed_functions).extract(
            base_prompt
        )

    @staticmethod
    def make_function_prompt(prompt: str, functions: List[Function]) -> str:
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
        return [json.dumps(f.name) for f in functions]
