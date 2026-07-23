from typing import List
import json
from .input_reader import Prompt, Function
from .state import TokenParser
from .prompt_state import PromptState
from .llm import LLM


class GenerationPipeline:
    llm: LLM
    prompt_state: PromptState
    token_parser: TokenParser
    response_found: bool

    def __init__(self, functions: List[Function]) -> None:
        self.llm = LLM()
        self.token_parser = TokenParser(self.llm)
        self.prompt_state = PromptState(functions)
        self.response_found = False

    def generate_prompt_response(self, prompt: Prompt) -> None:
        base_prompt: str = GenerationPipeline.format_function_prompt(
            prompt.prompt, self.prompt_state.functions
        )
        while True:
            # print(f"PROMT IS {base_prompt + self.prompt_state.generated_tokens}")
            input_ids = self.llm.encode(  # Tokenize prompt and get token ids
                base_prompt + self.prompt_state.generated_tokens
            )
            self.prompt_state.logits = self.llm.get_logits(input_ids)
            self.mask_logits()
            self.update_generated_tokens()
            print(f"generated so far: {self.prompt_state.generated_tokens}")
            print(self.token_parser.current_state)
            if self.response_found:
                break

        self.prompt_state.generated_tokens = '"'
        self.response_found = False

        base_prompt = GenerationPipeline.format_param_prompt(
            prompt.prompt, self.token_parser.current_function
        )

        while True:
            print(f"=====================\n{base_prompt + self.prompt_state.generated_tokens}\n=====================")
            input_ids = self.llm.encode(  # Tokenize prompt and get token ids
                base_prompt + self.prompt_state.generated_tokens
            )
            self.prompt_state.logits = self.llm.get_logits(input_ids)
            self.mask_logits()
            self.update_generated_tokens()
            print(f"generated so far: {self.prompt_state.generated_tokens}")
            print(self.token_parser.current_state)
            if self.response_found:
                break

        # base_prompt: str = GenerationPipeline.format_prompt(
        #     prompt.prompt, self.prompt_state.functions
        # )

        # while True:
        #     input_ids = self.llm.encode(  # Tokenize prompt and get token ids
        #         base_prompt + self.prompt_state.generated_tokens
        #     )
        #     self.prompt_state.logits = self.llm.get_logits(input_ids)
        #     self.mask_logits()
        #     self.update_generated_tokens()
        #     print(f"generated so far: {self.prompt_state.generated_tokens}")
        #     print(self.token_parser.current_state)
        #     if self.response_found:
        #         break

        return

    @staticmethod
    def format_function_prompt(prompt: str, functions: List[Function]) -> str:
        return (
            "For the given prompt, reply in JSON string with the function " +
            'name\n\n' +
            "Examples:\n" +
            'What is the sum of 100 and 50? → "fn_add_numbers"\n' +
            'Greet gandalf → "fn_greet"\n' +
            'Replace all vowels in "hello" with asterisks → ' +
            '"fn_substitute_string_with_regex"\n' +
            f"\n\nThe following functions are supported: {json.dumps([
                {"name": f.name, "description": f.description}
                for f in functions
            ])}" +
            f'\n\nPrompt: {prompt}\nOutput: '
        )

    @staticmethod
    def format_param_prompt(prompt: str, function: Function) -> str:
        params_str = ("\n").join([
            f"{json.dumps(p.name)}=<{p.type}>"
            for p in function.parameters
        ])

        params = [
            (p.name, p.type)
            for p in function.parameters
        ]
        print(f"params is {params}")

        param = params[0]
        return (
            "System: Output matching this schema:\n"
            f'"function_name"={json.dumps(function.name)}\n' +
            f"{params_str}\n" +
            "\n" +
            f"User: {prompt}\n"
            f"Output:\n" +
            f'"function_name"={json.dumps(function.name)}\n' +
            f"{json.dumps(param[0])}="
        )

    @staticmethod
    def format_prompt(prompt: str, functions: List[Function]) -> str:
        return (
            "For the given prompt, reply in strict JSON with the format " +
            "[function_name,parameter1_name,parameter1_value," +
            "parameter2_name,parameter2_value,...]\n\n" +
            "Examples:\n" +
            "What is the sum of 100 and 50? → a=100.0 b=50.0 → " +
            '["fn_add_numbers","a",100.0,"b",50.0]\n' +
            'Greet alice → name=alice → ["fn_greet","name","alice"]\n' +
            'Replace all vowels in "hello" with asterisks → ' +
            'source_string=hello regex=[aeiou] replacement=* → ' +
            '["fn_substitute_string_with_regex","source_string",' +
            '"hello","regex","[aeiou]","replacement","*"]\n' +
            "Note: Extract the actual values from the prompt - do not use " +
            "the example values.\n" +
            "Note 2: Integers must be written as floats e.g. 44 → 44.0\n" +
            "Note 3: Separate each value with a comma immediately " +
            "after, except the last one\n" +
            f"\n\nThe following functions are supported: {json.dumps(
                [f.model_dump() for f in functions]
            )}" +
            f"\n\nPrompt: {prompt}\nOutput: "
        )

    def mask_logits(self) -> None:
        self.token_parser.parse_token(self.prompt_state)

    def update_generated_tokens(self) -> None:
        next_id = max(self.prompt_state.logits)  # WHAT IF MORE THAN 1 MAX????
        bal = self.prompt_state.logits.index(next_id)
        next_token = self.llm.decode(bal)
        self.prompt_state.generated_tokens += next_token
        self.response_found = self.token_parser.update_current_state(
            self.prompt_state.generated_tokens, self.prompt_state.functions
        )

    # @staticmethod
    # def apply_constrained_formula(valid_ids: Dict[int, int]) -> None:  # Calculate ratios
    #     pass

    # @staticmethod
    # def choose_next_token(valid_ids: Dict[int, int]) -> int:
    #     pass

    # @staticmethod
    # def check_if_end_json(generated_tokens: str) -> bool:  # Check if a block is closed
    #     pass


# A Tensor is essentially an n-dimensional array, but it lives on a specific device (CPU, GPU, or in your case Apple's MPS — Metal Performance Shaders). NumPy only knows how to work with memory on the CPU/host RAM.
# When your tensor is on MPS, its data physically lives in GPU memory. NumPy can't reach in there directly — that's why the conversion fails.

# A rough rule of thumb for English: 1 word ≈ 1.3–1.5 tokens.

# json.dumps(result, indent=2) for print output in JSON

# State.EXPECT_OPEN_BRACKET  → generated_tokens matches r'^\[$'
# State.EXPECT_NAME          → generated_tokens matches r'^\["[^"]*"?$'
# State.EXPECT_COMMA         → generated_tokens matches r'^\["[^"]*",$'
# State.EXPECT_PARAMETERS    → generated_tokens matches r'^\["[^"]*",\s*\{.*$'
# State.EXPECT_CLOSE_BRACKET → generated_tokens matches r'^\["[^"]*",\s*\{.*\}$'
# State.DONE                 → generated_tokens matches r'^\["[^"]*",\s*\{.*\}\]$'
