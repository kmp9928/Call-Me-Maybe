import sys
from typing import List
from .errors import (
    CommandLineArgumentsError,
    InputFileError,
    JSONExtractorError
)
from .input_reader import InputReader
from .function_name_extractor import FunctionNameExtractor
from .llm import LLM
from .models import PromptResponse
from .output_writer import OutputWriter
from .parameters_extractor import ParametersExtractor
from .prompt_pipeline import PromptPipeline


if __name__ == "__main__":
    try:
        args = InputReader()
        functions = args.get_functions()
        prompts = args.get_prompts()

        llm = LLM()
        function_name_extractor = FunctionNameExtractor(llm)
        parameters_extractor = ParametersExtractor(llm)
        pipeline = PromptPipeline(
            functions, function_name_extractor, parameters_extractor
        )

        responses: List[PromptResponse] = []
        for n, prompt in enumerate(prompts, start=1):
            print(f"Processing prompt {n}/{len(prompts)}")
            response = pipeline.process_prompt(prompt)
            # debug("output", response)
            # break
            responses.append(response)

        OutputWriter().write_output(args.get_output_file(), responses)
    except CommandLineArgumentsError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except (InputFileError, JSONExtractorError) as e:
        print(f"Error: {e}")
