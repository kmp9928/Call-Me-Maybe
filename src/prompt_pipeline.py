from typing import Any, Dict, List
from .function_name_extractor import FunctionNameExtractor
from .parameters_extractor import ParametersExtractor
from .models import Function, Prompt, PromptResponse


class PromptPipeline:
    functions: List[Function]
    function_name_extractor: FunctionNameExtractor
    parameters_extractor: ParametersExtractor

    def __init__(
        self,
        functions: List[Function],
        function_name_extractor: FunctionNameExtractor,
        parameters_extractor: ParametersExtractor
    ) -> None:
        self.functions = functions
        self.function_name_extractor = function_name_extractor
        self.parameters_extractor = parameters_extractor

    def process_prompt(self, prompt: Prompt) -> PromptResponse:
        function: Function = self.get_function(prompt)
        parameters: Dict[str, Any] = self.get_parameters(prompt, function)

        return PromptResponse(
            prompt=prompt.prompt, function=function, parameters=parameters
        )

    def get_function(self, prompt: Prompt) -> Function:
        function_name: str = self.function_name_extractor.extract(
            prompt, self.functions
        )

        return next(
            f for f in self.functions if f.name == function_name
        )

    def get_parameters(
        self, prompt: Prompt, function: Function
    ) -> Dict[str, Any]:
        return self.parameters_extractor.extract(prompt, function)
