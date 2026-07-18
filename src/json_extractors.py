import json
import re
from abc import ABC, abstractmethod
from typing import List, Any, Protocol
from .llm import LLM
from .debug import debug

MIN_INF = float('-inf')
PARTIAL_STR_REGEX = re.compile(
    r'^"(([^"\\\n\t\r]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{0,4})+"?)?\Z'
)
STR_REGEX = re.compile(r'^"([^"\\\n\t\r]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})+"\Z')
PARTIAL_NUMBER_REGEX = re.compile(r'^-?\d+\.?\d{0,6}\Z')
NUMBER_REGEX = re.compile(r'^-?\d+\.\d{6}\Z')
PARTIAL_REGEX = re.compile(
    r'^"((\\\\[b]([^"\\]|\\.)*|\[[^\]"]*(\][+*]?)?)"?)?\Z'

)
REGEX = re.compile(r'^"(\\\\[b]([^"\\]|\\.)+|\[[^\]"]+\][+*]?)"\Z')


class PromptRulesProvider(Protocol):
    def get_rules(self) -> str:
        ...


class JSONExtractor(ABC):
    llm: LLM

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def extract(self, base_prompt: str) -> Any:
        output = self.get_output_prefix()

        while not self.is_valid_output(output):
            debug("base_prompt with output", base_prompt + output)
            input_ids = self.llm.encode(base_prompt + output)
            logits = self.llm.get_logits(input_ids)
            for id in range(len(logits)):
                token: str = self.llm.decode(id)
                if len(token) == 0 or not self.is_valid_token(output, token):
                    logits[id] = MIN_INF

            best_id = max(logits)
            output += self.llm.decode(logits.index(best_id))

        debug("output in extract", output)
        return json.loads(output)

    def get_output_prefix(self) -> str:
        return ""

    def get_rules(self) -> str:
        return ""

    @abstractmethod
    def is_valid_token(self, output: str, token: str) -> bool:
        pass

    @abstractmethod
    def is_valid_output(self, output: str) -> bool:
        pass


class LiteralJSONExtractor(JSONExtractor):
    json_literals: List[str]

    def __init__(self, llm: LLM, json_literals: List[str]) -> None:
        super().__init__(llm)
        self.json_literals = json_literals

    def is_valid_token(self, output: str, token: str) -> bool:
        current = output + token
        for literal in self.json_literals:
            if literal.startswith(current):
                return True

        return False

    def is_valid_output(self, output: str) -> bool:
        return output in self.json_literals


class BooleanJSONExtractor(LiteralJSONExtractor):
    def __init__(self, llm: LLM) -> None:
        super().__init__(llm, ["true", "false"])


class StringJSONExtractor(JSONExtractor):
    def get_output_prefix(self) -> str:
        return '"'

    def is_valid_token(self, output: str, token: str) -> bool:
        match = re.match(PARTIAL_STR_REGEX, output + token)
        return True if match is not None else False

    def is_valid_output(self, output: str) -> bool:
        match = re.match(STR_REGEX, output)
        return True if match is not None else False


class RegexJSONExtractor(JSONExtractor):
    def get_rules(self) -> str:
        return (
            "Don't extract the regex values from the prompt. " +
            "Always use proper regex sets " +
            r'(e.g. "\\bword\\b", "[aeiouAEIOU]", "[0-9]+").' +
            "\n" +
            "Example: " +
            'Substitute the word "pencil" with "pen" in "I love my pencil" ' +
            "source_string='The cat sat on the mat with another cat' " +
            r'regex="\\bpencil\\b" ' +
            'replacement="pen"'
        )

    def is_valid_token(self, output: str, token: str) -> bool:
        match = re.match(PARTIAL_REGEX, output + token)
        return True if match is not None else False

    def is_valid_output(self, output: str) -> bool:
        match = re.match(REGEX, output)
        return True if match is not None else False


class NumberJSONExtractor(JSONExtractor):
    def get_rules(self) -> str:
        return "Extract the actual number values from the prompt as floats."

    def is_valid_token(self, output: str, token: str) -> bool:
        match = re.match(PARTIAL_NUMBER_REGEX, output + token)
        return True if match is not None else False

    def is_valid_output(self, output: str) -> bool:
        match = re.match(NUMBER_REGEX, output)
        return True if match is not None else False
