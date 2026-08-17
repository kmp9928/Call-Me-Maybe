import copy
import json
import re
from typing import Dict, Any
from .errors import JSONExtractorTypeError, JSONExtractorParsingError
from .json_extractors import (
    PromptRulesProvider,
    JSONExtractor,
    BooleanJSONExtractor,
    StringJSONExtractor,
    RegexJSONExtractor,
    ReplacementJSONExtractor,
    PathJSONExtractor,
    NumberJSONExtractor,
    IntegerJSONExtractor
)
from .llm import LLM
from .models import Function, Parameter, Prompt


class ParametersExtractor:
    """Extracts typed parameter values for a function call using LLMs.

    Attributes:
        llm (LLM): Wrapper instance used for token-level generation.
    """

    llm: LLM

    def __init__(self, llm: LLM) -> None:
        """Initializes the parameter extractor.

        Args:
            llm (LLM): Language model wrapper instance.
        """
        self.llm = llm

    def extract(self, prompt: Prompt, function: Function) -> Dict[str, Any]:
        """Extracts required parameter values for a specified function.

        Args:
            prompt (Prompt): User prompt container object.
            function (Function): Target function specification.

        Returns:
            Dict[str, Any]: Mapping of parameter names to values.

        Raises:
            JSONExtractorParsingError: If extracted JSON cannot be
                parsed for a specific parameter.
        """
        parameter_values: Dict[str, Any] = {}

        remaining_prompt = prompt.prompt
        for param in function.parameters:
            extractor = self.choose_extractor(param)
            base_prompt = ParametersExtractor.make_parameters_prompt(
                prompt.prompt, function, param, parameter_values, extractor
            )
            try:
                parameter_values[param.name] = extractor.extract(
                    base_prompt, remaining_prompt
                )
                if param.type == "number" or param.type == "integer":
                    remaining_prompt = ParametersExtractor.update_prompt(
                        remaining_prompt, parameter_values[param.name]
                    )
            except json.JSONDecodeError:
                raise JSONExtractorParsingError(param.name)

        return parameter_values

    def choose_extractor(self, parameter: Parameter) -> JSONExtractor:
        """Selects the extractor strategy based on parameter attributes.

        Args:
            parameter (Parameter): Parameter definition object.

        Returns:
            JSONExtractor: Specialized extractor instance.

        Raises:
            JSONExtractorTypeError: If parameter type is unsupported.
        """
        if parameter.type == "number":
            return NumberJSONExtractor(self.llm)
        elif parameter.type == "integer":
            return IntegerJSONExtractor(self.llm)
        elif parameter.type == "string":
            if parameter.name == "regex":
                return RegexJSONExtractor(self.llm)
            elif parameter.name == "replacement":
                return ReplacementJSONExtractor(self.llm)
            elif parameter.name == "path":
                return PathJSONExtractor(self.llm)
            return StringJSONExtractor(self.llm)
        elif parameter.type == "boolean":
            return BooleanJSONExtractor(self.llm)
        else:
            raise JSONExtractorTypeError(parameter.type)

    @staticmethod
    def make_parameters_prompt(
        prompt: str,
        function: Function,
        current_parameter: Parameter,
        parameter_values: Dict[str, Any],
        rules_provider: PromptRulesProvider
    ) -> str:
        """Builds system and user prompt for parameter extraction.

        Args:
            prompt (str): Raw user prompt string.
            function (Function): Target function specification.
            current_parameter (Parameter): Active parameter to extract.
            parameter_values (Dict[str, Any]): Extracted parameters.
            rules_provider (PromptRulesProvider): Provider for rules.

        Returns:
            str: Formatted prompt string for the model.
        """
        rules = rules_provider.get_rules()
        rules = f"Rules: {rules}\n\n" if rules != "" else ""
        if current_parameter.name == "replacement":
            prompt = ParametersExtractor.simplify_prompt(prompt)

        return (
            "System: Output matching this schema:\n" +
            ParametersExtractor.make_schema(function) +
            "\n\n" +
            rules +
            f"User Prompt: {prompt}\n"
            f"Output:\n" +
            ParametersExtractor.make_output(
                function, current_parameter, parameter_values
            )
        )

    @staticmethod
    def make_schema(function: Function) -> str:
        """Generates a JSON schema representation of the function call.

        Args:
            function (Function): Function metadata.

        Returns:
            str: JSON-encoded schema string.
        """
        schema: Dict[str, Any] = {
            "function_name": function.name,
            "parameters": {}
        }
        for p in function.parameters:
            schema["parameters"][p.name] = (
                ParametersExtractor.format_parameter_placeholder(p)
            )

        return f"{json.dumps(schema)}"

    @staticmethod
    def make_output(
        function: Function,
        current_parameter: Parameter,
        parameter_values: Dict[str, Any]
    ) -> str:
        """Constructs partial JSON output sequence to guide model output.

        Args:
            function (Function): Target function metadata.
            current_parameter (Parameter): Current target parameter.
            parameter_values (Dict[str, Any]): Previously extracted values.

        Returns:
            str: Truncated JSON string prefix.
        """
        values = copy.deepcopy(parameter_values)
        values[current_parameter.name] = None

        return json.dumps({
            "function_name": function.name,
            "parameters": values
        }).removesuffix("null}}")

    @staticmethod
    def format_parameter_placeholder(parameter: Parameter) -> str:
        """Formats a type placeholder string for schema generation.

        Args:
            parameter (Parameter): Target parameter object.

        Returns:
            str: Type placeholder string enclosed in angle brackets.
        """
        return f"<{parameter.type}>"

    @staticmethod
    def simplify_prompt(prompt: str) -> str:
        """Normalizes single quotes to brackets in the prompt to prevent syntax
        errors.

        Args:
            prompt (str): Raw input prompt text.

        Returns:
            str: Sanitized prompt string.
        """
        pattern = r"(?<!\w)'(.*?)\'(?!\w)"

        return re.sub(pattern, r'[\1]', prompt)

    @staticmethod
    def update_prompt(user_prompt: str, parameter_value: int | float) -> str:
        if parameter_value % 1 == 0:
            str_parameter = str(int(parameter_value))
        else:
            str_parameter = str(parameter_value)

        return user_prompt.replace(str_parameter, "", 1)
