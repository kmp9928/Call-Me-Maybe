from typing import Any, Dict, List
from .function_name_extractor import FunctionNameExtractor
from .parameters_extractor import ParametersExtractor
from .models import Function, Prompt, PromptResponse


class PromptPipeline:
    """Executes prompt processing to identify functions and parameters.

    Attributes:
        functions (List[Function]): Supported function specifications.
        function_name_extractor (FunctionNameExtractor): Extractor for
            identifying target functions.
        parameters_extractor (ParametersExtractor): Extractor for
            parsing function parameters.
    """

    functions: List[Function]
    function_name_extractor: FunctionNameExtractor
    parameters_extractor: ParametersExtractor

    def __init__(
        self,
        functions: List[Function],
        function_name_extractor: FunctionNameExtractor,
        parameters_extractor: ParametersExtractor
    ) -> None:
        """Initializes PromptPipeline with functions and extractors.

        Args:
            functions (List[Function]): List of candidate functions.
            function_name_extractor (FunctionNameExtractor): Extractor
                instance for function selection.
            parameters_extractor (ParametersExtractor): Extractor
                instance for parameter resolution.
        """
        self.functions = functions
        self.function_name_extractor = function_name_extractor
        self.parameters_extractor = parameters_extractor

    def process_prompt(self, prompt: Prompt) -> PromptResponse:
        """Processes a prompt into a complete function call response.

        Args:
            prompt (Prompt): Raw prompt object to process.

        Returns:
            PromptResponse: Resolved function and extracted parameters.
        """
        function: Function = self.get_function(prompt)
        parameters: Dict[str, Any] = self.get_parameters(prompt, function)

        return PromptResponse(
            prompt=prompt.prompt, function=function, parameters=parameters
        )

    def get_function(self, prompt: Prompt) -> Function:
        """Identifies and resolves the matching function for a prompt.

        Args:
            prompt (Prompt): Raw prompt object.

        Returns:
            Function: Matched function specification.
        """
        function_name: str = self.function_name_extractor.extract(
            prompt, self.functions
        )

        return next(
            f for f in self.functions if f.name == function_name
        )

    def get_parameters(
        self, prompt: Prompt, function: Function
    ) -> Dict[str, Any]:
        """Extracts argument values required by the resolved function.

        Args:
            prompt (Prompt): Raw prompt object.
            function (Function): Target function specification.

        Returns:
            Dict[str, Any]: Extracted parameter names and values.
        """
        return self.parameters_extractor.extract(prompt, function)
