import json
import numpy as np
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Protocol
from .errors import JSONExtractorTimeoutError, JSONExtractorMissingParamError
from .llm import LLM


MIN_INF = float('-inf')
PARTIAL_STR_REGEX = re.compile(
    r'^"(([^"\\\n\t\r]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{0,4})+"?)?\Z'
)
STR_REGEX = re.compile(r'^"([^"\\\n\t\r]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})+"\Z')
PARTIAL_NUMBER_REGEX = re.compile(r'^"-?\d+\.?\d{0,6}"?\Z')
NUMBER_REGEX = re.compile(r'^"-?\d+\.?\d{0,6}"\Z')
PARTIAL_INTEGER_REGEX = re.compile(r'^"-?\d+"?\Z')
INTEGER_REGEX = re.compile(r'^"-?\d+"\Z')
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
    r'^"([a-zA-Z0-9_]+|\\["\\]|([^"\\a-zA-Z0-9_\s])\2{0,6})"?\Z'
)
REPLACEMENT_REGEX = re.compile(
    r'^"([a-zA-Z0-9_]+|\\["\\]|([^"\\a-zA-Z0-9_\s])\2{0,6})"\Z'
)
UNIX_PATH_REGEX = re.compile(r'/([^/ ]+/)*[^/ ]+')
HIGH_RISK_SYMBOL_MAP: Dict[str, str] = {
    "hyphen": "-",
    "hyphens": "-",
    "Pound": "#",
    "pounds": "#",
    "percent": "%",
    "at": "@",
    "ats": "@",
    "colons": ":",
    "semicolon": ";",
    "backslash": "\\",
    "pipe": "|",
    "equals": "=",
    "underscore": "_",
    "questionmark": "?",
    "questionmarks": "?"
}


class PromptRulesProvider(Protocol):
    """Protocol defining extraction rules."""

    def get_rules(self) -> str:
        """Returns the rule description string for prompt generation.

        Returns:
            str: Extraction rules formatted for inclusion in LLM prompts.
        """
        ...


class JSONExtractor(ABC):
    """Abstract base class for guided JSON extraction using an LLM.

    Attributes:
        llm (LLM): Wrapper instance for token encoding, decoding and logit
            retrieval.
    """

    llm: LLM

    def __init__(self, llm: LLM) -> None:
        """Initializes JSONExtractor with the target LLM.

        Args:
            llm (LLM): Model instance used for guided generation.
        """
        self.llm = llm

    def extract(self, base_prompt: str, user_prompt: str) -> Any:
        """Generates valid JSON by decoding tokens under grammar/regex
        constraints.

        Args:
            base_prompt (str): System/task prompt instructions.
            user_prompt (str): User input context for extraction.

        Returns:
            Any: The finalized, parsed JSON object or primitive value.

        Raises:
            JSONExtractorTimeoutError: If generated token sequence exceeds
                maximum length limit.
            JSONExtractorMissingParamError: If all candidate token logits are
                masked out.
        """
        output = self.get_output_prefix(user_prompt)

        while not self.is_valid_output(output):
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

            if np.isneginf(logits).all():
                raise JSONExtractorMissingParamError(user_prompt)
            best_id = max(logits)
            output += self.llm.decode(logits.index(best_id))

        return self.finalize_output(output)

    def get_output_prefix(self, user_prompt: str) -> str:
        """Returns initial string prefix to prepended to output generation.

        Args:
            user_prompt (str): Input prompt context.

        Returns:
            str: Initial output prefix string.
        """
        return ""

    def get_rules(self) -> str:
        """Returns extraction rules for prompt construction.

        Returns:
            str: Rule description string.
        """
        return ""

    @abstractmethod
    def is_valid_token(
        self, output: str, token: str, user_prompt: str
    ) -> bool:
        """Determines whether appending `token` to `output` remains
        syntactically valid.

        Args:
            output (str): Output string generated so far.
            token (str): Candidate token to evaluate.
            user_prompt (str): Input context prompt.

        Returns:
            bool: True if candidate token sequence is valid, False otherwise.
        """
        pass

    @abstractmethod
    def is_valid_output(self, output: str) -> bool:
        """Checks if the complete output sequence forms a valid result.

        Args:
            output (str): Output string generated so far.

        Returns:
            bool: True if output satisfies parsing criteria, False otherwise.
        """
        pass

    def finalize_output(self, output: str) -> Any:
        """Parses and formats the completed output string into a Python object.

        Args:
            output (str): Raw string output accumulated during extraction.

        Returns:
            Any: Parsed JSON object or primitive value.
        """
        return json.loads(output)


class LiteralJSONExtractor(JSONExtractor):
    """Constrains extraction strictly to a predefined list of options.

    Attributes:
        json_literals (List[str]): List of allowed values for generation.
    """

    json_literals: List[str]

    def __init__(self, llm: LLM, json_literals: List[str]) -> None:
        """Initializes LiteralJSONExtractor with allowed literal options.

        Args:
            llm (LLM): Model instance used for generation.
            json_literals (List[str]): Allowed exact string literals to match.
        """
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
    """Constrains extraction to boolean values (`'true'` or `'false'`)."""

    def __init__(self, llm: LLM) -> None:
        """Initializes BooleanJSONExtractor with 'true' and 'false' literals.

        Args:
            llm (LLM): Model instance used for generation.
        """
        super().__init__(llm, ["true", "false"])


class StringJSONExtractor(JSONExtractor):
    """Extracts string values that must originate from the user prompt."""

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
            value = json.loads(output + token)
        except json.JSONDecodeError:
            try:
                value = json.loads(output + token + '"')
            except json.JSONDecodeError:
                return True

        assert isinstance(value, str)
        if value not in user_prompt:
            return False

        return True

    def is_valid_output(self, output: str) -> bool:
        match = re.match(STR_REGEX, output)
        return True if match is not None else False

    def finalize_output(self, output: str) -> Any:
        final_str = json.loads(output)
        assert isinstance(final_str, str)

        return final_str.strip()


class PathJSONExtractor(StringJSONExtractor):
    """Extracts absolute file path strings from user prompts."""

    def get_output_prefix(self, user_prompt: str) -> str:
        if re.search(UNIX_PATH_REGEX, user_prompt) is not None:
            return '"/'

        return super().get_output_prefix(user_prompt)

    def get_rules(self) -> str:
        return "Extract the absolute path."


class ReplacementJSONExtractor(StringJSONExtractor):
    """Extracts replacement strings or single-character non-word symbols."""

    def get_rules(self) -> str:
        return (
            "For a non-word symbol, output one character. " +
            "asterisks → '*'."
        )

    def is_valid_token(
        self, output: str, token: str, user_prompt: str
    ) -> bool:
        match = re.match(PARTIAL_REPLACEMENT_REGEX, output + token)
        return True if match is not None else False

    def is_valid_output(self, output: str) -> bool:
        match = re.match(REPLACEMENT_REGEX, output)
        return True if match is not None else False

    def finalize_output(self, output: str) -> Any:
        value = super().finalize_output(output)
        assert isinstance(value, str)

        if len(value) > 1 and len(set(value)) == 1:
            return value[0]

        if value in HIGH_RISK_SYMBOL_MAP.keys():
            return HIGH_RISK_SYMBOL_MAP[value]

        return value


class RegexJSONExtractor(JSONExtractor):
    """Extracts and formats regular expression patterns from user input."""

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
    """Extracts floating-point numeric values from user prompts."""

    def is_valid_token(
        self, output: str, token: str, user_prompt: str
    ) -> bool:
        if re.match(PARTIAL_NUMBER_REGEX, output + token) is None:
            return False

        try:
            value = json.loads(output + token)
        except json.JSONDecodeError:
            try:
                value = json.loads(output + token + '"')
            except json.JSONDecodeError:
                return True

        assert isinstance(value, str)
        if value not in user_prompt:
            return False

        return True

    def is_valid_output(self, output: str) -> bool:
        match = re.match(NUMBER_REGEX, output)
        return True if match is not None else False

    def finalize_output(self, output: str) -> Any:
        return float(output.strip('"'))


class IntegerJSONExtractor(StringJSONExtractor):
    """Extracts integer values from user prompts."""

    def is_valid_token(
        self, output: str, token: str, user_prompt: str
    ) -> bool:
        if re.match(PARTIAL_INTEGER_REGEX, output + token) is None:
            return False

        try:
            value = json.loads(output + token)
        except json.JSONDecodeError:
            try:
                value = json.loads(output + token + '"')
            except json.JSONDecodeError:
                return True

        assert isinstance(value, str)
        if value not in user_prompt:
            return False

        return True

    def is_valid_output(self, output: str) -> bool:
        match = re.match(INTEGER_REGEX, output)
        return True if match is not None else False

    def finalize_output(self, output: str) -> Any:
        return int(output.strip('"'))
