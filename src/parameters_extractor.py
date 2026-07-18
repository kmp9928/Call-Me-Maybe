import json
from typing import Dict, Any
from .errors import JSONExtractorError
from .json_extractors import (
    JSONExtractor,
    PromptRulesProvider,
    StringJSONExtractor,
    NumberJSONExtractor,
    RegexJSONExtractor
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

        for parameter in function.parameters:
            debug("in parameter", parameter)
            extractor = self.choose_extractor(parameter)
            base_prompt = ParametersExtractor.make_parameters_prompt(
                prompt.prompt, function, parameter, parameter_values, extractor
            )
            parameter_values[parameter.name] = extractor.extract(base_prompt)

        return parameter_values

    def choose_extractor(self, parameter: Parameter) -> JSONExtractor:
        if parameter.type.type == "number":
            return NumberJSONExtractor(self.llm)
        elif parameter.type.type == "string":
            if parameter.name == "regex":
                return RegexJSONExtractor(self.llm)
            return StringJSONExtractor(self.llm)
        else:
            raise JSONExtractorError(
                f"Unsupported parameter type {parameter.type.type}"
            )

    @staticmethod
    def make_parameters_prompt(
        prompt: str,
        function: Function,
        current_parameter: Parameter,
        parameter_values: Dict[str, Any],
        rules_provider: PromptRulesProvider
    ) -> str:
        rules = rules_provider.get_rules()
        rules = f"Rules: {rules}\n" if rules != "" else ""

        return (
            "System: Output matching this schema:\n" +
            ParametersExtractor.make_schema(function) +
            "\n" +
            rules +
            f"User: {prompt}\n"
            f"Output:\n" +
            ParametersExtractor.make_output(
                function, current_parameter, parameter_values
            )
        )

    @staticmethod
    def make_schema(function: Function) -> str:
        parameters_str = ("\n").join([
            f"{json.dumps(p.name)}=<{p.type.type}>"
            for p in function.parameters
        ])

        return (
            f'"function_name"={json.dumps(function.name)}\n' +
            f"{parameters_str}\n"
        )

    @staticmethod
    def make_output(
        function: Function,
        current_parameter: Parameter,
        parameter_values: Dict[str, Any]
    ) -> str:
        output = f'"function_name"={json.dumps(function.name)}\n'
        for k, v in parameter_values.items():
            output += f"{json.dumps(k)}={json.dumps(v)}\n"

        return output + f"{json.dumps(current_parameter.name)}="
