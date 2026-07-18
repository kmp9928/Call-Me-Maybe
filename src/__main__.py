import sys
from .arguments_parser import ArgumentsParser
from .errors import InputFileError
from .function_name_extractor import FunctionNameExtractor
from .parameters_extractor import ParametersExtractor
from .llm import LLM
from .prompt_pipeline import PromptPipeline
from .debug import debug

if __name__ == "__main__":
    try:
        args = ArgumentsParser()
        functions = args.get_functions()
        prompts = args.get_prompts()

        llm = LLM()
        function_name_extractor = FunctionNameExtractor(llm)
        parameters_extractor = ParametersExtractor(llm)
        pipeline = PromptPipeline(
            functions, function_name_extractor, parameters_extractor
        )
        for prompt in prompts:
            response = pipeline.process_prompt(prompt)
            debug("response", response)
            break
    except SystemExit as e:
        print(f"Error: Missing required argument '--{str(e).split("--")[1]}'")
        sys.exit(1)  # Not strictly needed, but without it the program would continue executing after the except block
    except InputFileError as e:
        print(f"Error: {e}")
