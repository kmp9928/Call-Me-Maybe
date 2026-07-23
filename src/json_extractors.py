import json
import re
from abc import ABC, abstractmethod
from typing import List, Any, Protocol
from .errors import JSONExtractorTimeoutError
from .llm import LLM
from .debug import debug


MIN_INF = float('-inf')
PARTIAL_STR_REGEX = re.compile(
    r'^"(([^"\\\n\t\r]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{0,4})+"?)?\Z'
)
STR_REGEX = re.compile(r'^"([^"\\\n\t\r]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})+"\Z')

PARTIAL_INTEGER_REGEX = re.compile(r'^-?[0-9]+\Z')
INTEGER_REGEX = re.compile(r'^-?[0-9]+\Z')
PARTIAL_NUMBER_REGEX = re.compile(r'^"-?\d+\.?\d{0,6}"?\Z')
NUMBER_REGEX = re.compile(r'^"-?\d+\.?\d{0,6}"\Z')

PARTIAL_REGEX = re.compile(
    r'^"('
    r'\[(B(\])?)?[^\]]*(\[B\]?)?|'
    r'\[(0(-(9)?)?)?\]?(\+)?|'
    r'\[[aeiouAEIOU]+\]?'
    r')?"?\Z'
)
REGEX = re.compile(
    r'^"(\[B\][^\]]+\[B\]|\[0-9\]\+|\[[aeiouAEIOU]+\])"\Z'
)
PARTIAL_REPLACEMENT_REGEX = re.compile(
    r'^"([a-zA-Z0-9_-]+|\\["\\]|[^"\\a-zA-Z0-9_\s-])"?\Z'
)
REPLACEMENT_REGEX = re.compile(
    r'^"([a-zA-Z0-9_-]+|\\["\\]|[^"\\a-zA-Z0-9_\s-])"\Z'
)
UNIX_PATH_REGEX = re.compile(r'/([^/ ]+/)*[^/ ]+')


class PromptRulesProvider(Protocol):
    def get_rules(self) -> str:
        ...


class JSONExtractor(ABC):
    llm: LLM

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def extract(self, base_prompt: str, user_prompt: str) -> Any:
        output = self.get_output_prefix(user_prompt)

        while not self.is_valid_output(output):
            # debug("base_prompt + output", base_prompt + output)
            if len(base_prompt + output) >= len(base_prompt) + 150:
                raise JSONExtractorTimeoutError(output)

            input_ids = self.llm.encode(base_prompt + output)
            logits = self.llm.get_logits(input_ids)
            for id in range(len(logits)):
                token: str = self.llm.decode(id)
                if len(token) == 0 or not self.is_valid_token(
                    output, token, user_prompt
                ):
                    logits[id] = MIN_INF

            best_id = max(logits)
            output += self.llm.decode(logits.index(best_id))
            # debug("output so far", output)

        return self.finalize_output(output)

    def get_output_prefix(self, user_prompt: str) -> str:
        return ""

    def get_rules(self) -> str:
        return ""

    @abstractmethod
    def is_valid_token(
        self, output: str, token: str, user_prompt: str
    ) -> bool:
        pass

    @abstractmethod
    def is_valid_output(self, output: str) -> bool:
        pass

    def finalize_output(self, output: str) -> Any:
        return json.loads(output)


class LiteralJSONExtractor(JSONExtractor):
    json_literals: List[str]

    def __init__(self, llm: LLM, json_literals: List[str]) -> None:
        super().__init__(llm)
        self.json_literals = json_literals

    def is_valid_token(
        self, output: str, token: str, user_prompt: str
    ) -> bool:
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
    def get_output_prefix(self, user_prompt: str) -> str:
        return '"'

    def get_rules(self) -> str:
        return "Extract fully from User Prompt."

    def is_valid_token(
        self, output: str, token: str, user_prompt: str
    ) -> bool:
        if re.match(PARTIAL_STR_REGEX, output + token) is None:
            return False

        try:
            value = json.loads(output + token + '"')
            assert isinstance(value, str)
            if value not in user_prompt:
                return False
        except json.JSONDecodeError:
            pass

        return True

    def is_valid_output(self, output: str) -> bool:
        match = re.match(STR_REGEX, output)
        return True if match is not None else False


class PathJSONExtractor(StringJSONExtractor):
    def get_output_prefix(self, user_prompt: str) -> str:
        if re.search(UNIX_PATH_REGEX, user_prompt) is not None:
            return '"/'

        return super().get_output_prefix(user_prompt)

    def get_rules(self) -> str:
        return "Extract the absolute path."


class ReplacementJSONExtractor(StringJSONExtractor):
    def get_rules(self) -> str:
        return "If asked for a non-word character. (e.g. asterisks), output EXACTLY ONE character (e.g. '*')."

    def is_valid_token(
        self, output: str, token: str, user_prompt: str
    ) -> bool:
        match = re.match(PARTIAL_REPLACEMENT_REGEX, output + token)
        return True if match is not None else False

    def is_valid_output(self, output: str) -> bool:
        match = re.match(REPLACEMENT_REGEX, output)
        return True if match is not None else False

    def finalize_output(self, output: str) -> Any:
        value = json.loads(output)
        assert isinstance(value, str)
        if len(value) > 1 and len(set(value)) == 1:
            return value[0]

        return value


class RegexJSONExtractor(JSONExtractor):
    def get_rules(self) -> str:
        return (
            "Generate quoted regex string based on the input. " +
            'Words → "[B]word[B]" | Vowels → "[aeiouAEIOU]" | ' +
            'Digits → "[0-9]+".'
        )

    def is_valid_token(
        self, output: str, token: str, user_prompt: str
    ) -> bool:
        match = re.match(PARTIAL_REGEX, output + token)
        return True if match is not None else False

    def is_valid_output(self, output: str) -> bool:
        match = re.match(REGEX, output)
        return True if match is not None else False

    def finalize_output(self, output: str) -> Any:
        if output == '"[0-9]+"':
            return r'\d+'
        elif output.startswith('"[B]'):
            word = output.strip('"').replace("[B]", "")
            return rf'\b{re.escape(word)}\b'
        return json.loads(output)


class NumberJSONExtractor(StringJSONExtractor):
    def is_valid_token(
        self, output: str, token: str, user_prompt: str
    ) -> bool:
        match = re.match(PARTIAL_NUMBER_REGEX, output + token)
        return True if match is not None else False

    def is_valid_output(self, output: str) -> bool:
        match = re.match(NUMBER_REGEX, output)
        return True if match is not None else False

    def finalize_output(self, output: str) -> Any:
        return float(output.strip('"'))


class IntegerJSONExtractor(NumberJSONExtractor):
    def finalize_output(self, output: str) -> Any:
        return int(super().finalize_output(output))
