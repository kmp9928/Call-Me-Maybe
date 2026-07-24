import copy
import json
import re
from enum import Enum
from typing import Dict, Any
from .errors import JSONExtractorTypeError, JSONExtractorParsingError
from .json_extractors import (
    JSONExtractor,
    PathJSONExtractor,
    PromptRulesProvider,
    NumberJSONExtractor,
    IntegerJSONExtractor,
    BooleanJSONExtractor,
    StringJSONExtractor,
    RegexJSONExtractor,
    ReplacementJSONExtractor
)
from .llm import LLM
from .models import Function, Parameter, Prompt
from .debug import debug


class ParametersExtractor:
    llm: LLM

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def extract(self, prompt: Prompt, function: Function) -> Dict[str, Any]:
        parameter_values: Dict[str, Any] = {}

        for param in function.parameters:
            # debug("param", param)
            extractor = self.choose_extractor(param)
            base_prompt = ParametersExtractor.make_parameters_prompt(
                prompt.prompt, function, param, parameter_values, extractor
            )
            try:
                parameter_values[param.name] = extractor.extract(
                    base_prompt, prompt.prompt
                )
                # debug("is", parameter_values[param.name])
            except json.JSONDecodeError:
                raise JSONExtractorParsingError(param.name)

        return parameter_values

    def choose_extractor(self, parameter: Parameter) -> JSONExtractor:
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
        schema = {"function_name": function.name, "parameters": {}}
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
        values = copy.deepcopy(parameter_values)
        values[current_parameter.name] = None

        return json.dumps({
            "function_name": function.name,
            "parameters": values
        }).removesuffix("null}}")

    @staticmethod
    def format_parameter_placeholder(parameter: Parameter) -> str:
        return f"<{parameter.type}>"

    @staticmethod
    def simplify_prompt(prompt: str) -> str:
        #changes '' for [] to make the prompt easier to find symbols like "
        pattern = r"(?<!\w)'(.*?)\'(?!\w)"

        return re.sub(pattern, r'[\1]', prompt)
