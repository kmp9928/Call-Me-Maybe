from typing import List, Dict, Optional, Any
from enum import Enum
import re
import json
from .arguments_parser import Function
from .prompt_state import PromptState
from .llm import LLM
import math
import random

# import numpy as np


class State(Enum):
    EXPECT_OPEN_BRACKET = "expect_open_bracket"
    EXPECT_NAME = "expect_name"
    EXPECT_COMMA = "expect_comma"
    EXPECT_PARAMETERS = "expect_parameters"
    EXPECT_CLOSE_BRACKET = "expect_close_bracket"
    DONE = "done"


class TokenParser:
    llm: LLM
    states: Dict[State, str]
    current_state: State
    current_function: Optional[Function]
    current_parameters: List[str]
    parameters_regex: List[re.Pattern]
    parameters_index: int
    generated_tokens_index: int
    parameters_values: Dict[str, Any]

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.states = {
            # State.EXPECT_OPEN_BRACKET.value: self.expect_open_bracket,
            State.EXPECT_NAME.value: self.expect_name,
            # State.EXPECT_COMMA.value: self.expect_comma,
            State.EXPECT_PARAMETERS.value: self.expect_parameters,
            State.EXPECT_CLOSE_BRACKET.value: self.expect_close_bracket,
        }
        self.current_state = State.EXPECT_NAME.value
        self.current_function = None
        self.current_parameters = []
        self.parameters_regex = []
        self.parameters_index = 0
        self.generated_tokens_index = 0
        self.parameters_values = {}

    def parse_token(
        self,
        prompt_state: PromptState
    ) -> None:
        self.states[self.current_state](
            prompt_state.logits,
            prompt_state.generated_tokens,
            prompt_state.functions
        )
        # self.states[self.current_state](logits, generated_tokens, functions)

    # def expect_open_bracket(
    #     self,
    #     logits: List[float],
    #     _generated_tokens: str,
    #     functions: List[Function]
    # ) -> None:
    #     sorted_logits_ids: List[int] = [
    #         index
    #         for index, _ in sorted(
    #             enumerate(logits),
    #             key=lambda x: x[1],
    #             reverse=True
    #         )
    #     ]
    #     opts: List[str] = [
    #         '[' + json.dumps(function.name) for function in functions  # ADD ESCAPE????
    #     ]

    #     for id in sorted_logits_ids:
    #         text: str = self.llm.decode(id)
    #         if (
    #             len(text) == 0 or not TokenParser.is_valid(text, opts)
    #         ):
    #             logits[id] = float('-inf')
            # else:
            #     print(f"possible is {text} with logit {logits[id]}")

        # print(f"expect_open_bracket: {[self.llm.decode([i]) for i, logit in enumerate(logits) if logit != float('-inf')]}")

    def expect_name(  # CAPS RESULT TILL '"' -> IS THIS RIGHT???!!! PRO: FROM HERE FOR SURE GO TO FIND COMMA, F_NAME IS FOUND AND PARAMS TOO / CON: NOT PROPER BUFFER
        self,
        logits: List[float],
        generated_tokens: str,
        functions: List[Function]
    ) -> None:
        sorted_logits_ids: List[int] = [
            index
            for index, _ in sorted(
                enumerate(logits),
                key=lambda x: x[1],
                reverse=True
            )
        ]
        opts: List[str] = [
            json.dumps(function.name) for function in functions  # ADD ESCAPE????
        ]

        for id in sorted_logits_ids:
            text: str = self.llm.decode(id)
            if (
                len(text) == 0 or
                not TokenParser.is_valid(generated_tokens + text, opts)
            ):
                logits[id] = float('-inf')
            # else:
            #     print(f"possible is {text} with logit {logits[id]}")

        # print(f"expect_name: {[self.llm.decode([i]) for i, logit in enumerate(logits) if logit != float('-inf')]}")

    # def expect_comma(  # FORCES TOKEN TO BE ',' -> IS THIS RIGHT???!!!
    #     self,
    #     logits: List[float],
    #     generated_tokens: str,
    #     functions: List[Function]
    # ) -> None:
    #     sorted_logits_ids: List[int] = [
    #         index
    #         for index, _ in sorted(
    #             enumerate(logits),
    #             key=lambda x: x[1],
    #             reverse=True
    #         )
    #     ]
    #     self.current_function = next(
    #         (f for f in functions if f.name == generated_tokens[2:-1]), None
    #     )
    #     self.current_parameters = [
    #         value
    #         for p in self.current_function.parameters
    #         for value in (p.name, p.type.type)
    #     ]
    #     print(f"current_parameters is {self.current_parameters}")
    #     self.parameters_regex = TokenParser.generate_parameters_regex(
    #         self.current_parameters
    #     )
    #     print(f"param regex is {self.parameters_regex}")

    #     for id in sorted_logits_ids:
    #         text: str = self.llm.decode(id)
    #         if text != ',':
    #             logits[id] = float('-inf')

        # print(f"expect_name: {[self.llm.decode([i]) for i, logit in enumerate(logits) if logit != float('-inf')]}")

    def update_values(self, generated_tokens: str, functions: List[Function]) -> None:
        self.current_function = next(
            (f for f in functions if f.name == json.loads(generated_tokens)), None
        )
        # self.current_parameters = [
        #     value
        #     for p in self.current_function.parameters
        #     for value in (p.name, p.type.type)
        # ]
        self.current_parameters = [
            p.type.type
            for p in self.current_function.parameters
        ]
        print(f"current_parameters type are {self.current_parameters}")
        self.parameters_regex = TokenParser.generate_parameters_regex(
            self.current_parameters
        )
        print(f"param regex is {self.parameters_regex}")

    def expect_parameters(
        self,
        logits: List[float],
        generated_tokens: str,
        functions: List[Function]
    ) -> None:
        sorted_logits_ids: List[int] = [
            index
            for index, _ in sorted(
                enumerate(logits),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        # if self.generated_tokens_index == 0:
        #     self.generated_tokens_index += len(generated_tokens)

        # print(f"generated_tokens so far {generated_tokens[self.generated_tokens_index:]}")
        # print(f"found param {self.found_parameter(generated_tokens[self.generated_tokens_index:])}")

        if self.found_parameter(
            generated_tokens[self.generated_tokens_index:]
        ):
            self.generated_tokens_index = len(generated_tokens)
            assert self.current_function
            self.parameters_values[self.current_function.parameters[self.parameters_index].name] = json.loads(generated_tokens)
            self.parameters_index += 1
            return

        if self.parameters_index == len(self.current_parameters):
            print("DONE?")
            self.current_state = State.DONE.value
            return

        print(f"generated_tokens_index is {self.generated_tokens_index}")
        print(f"parameters_index is {self.parameters_index}")
        print(f"current param is {self.current_parameters[self.parameters_index]}")
        # print(f"will evalluate {generated_tokens[self.generated_tokens_index:]}")
        # parameters ["a", 2.0, "b", 3.0]
        # ["fn_add_numbers", "a", 2.0, "b", 3.0]

        for id in sorted_logits_ids:
            text: str = self.llm.decode(id)
            if (
                len(text) == 0 or
                not self.check_parameter_token(
                    generated_tokens[self.generated_tokens_index:], text
                )
            ):
                logits[id] = float('-inf')
            # else:
            #     print(f"possible is {text} with logit {logits[id]}")
        # if self.current_parameters[self.parameters_index] == "string":
        #     max_index = TokenParser.sample(logits)
        #     for index, logit in enumerate(logits):
        #         if index != max_index:
        #             logits[index] = float('-inf')

        # print(f"expect_name: {[self.llm.decode([i]) for i, logit in enumerate(logits) if logit != float('-inf')]}")

    def check_parameter_token(self, current: str, next: str) -> bool:
        # parameters ["a", "number", "b", "number"]
        if self.current_parameters[self.parameters_index] == "number":
            return TokenParser.is_valid_next_number_char(current, next)
            # return (False if re.match(r'-?\\d+\\.?\\d*', generated_parameters_tokens) is None else True)
        else:
            return TokenParser.is_valid_next_string_char(current, next)
            # return (False if re.match(r'"([^"\\]|\\.)+"', current + next) is None else True)

    @staticmethod
    def is_valid_next_number_char(current: str, next: str) -> bool:
        if "." in current:
            pattern = r'^-?\d+\.\d+$'
            return True if re.match(pattern, current + next) is not None else False
        else:
            return True if re.match(r'^-?\d+\.?\d*$', current + next) is not None else False

    @staticmethod
    def is_valid_next_string_char(current: str, next: str) -> bool:
        pattern = r'^"([^"\\]|\\.)*"?$'
        return True if re.match(pattern, current + next) is not None else False

    @staticmethod
    def sample(logits: list[float]) -> int:
        print("SAMPLING")
        temperature: float = 0.6
        scaled: List[float] = [logit / temperature for logit in logits]

        max_logit = max(scaled)
        exponential_logit = [math.exp(logit - max_logit) for logit in scaled]
        total = sum(exponential_logit)
        probabilities = [e / total for e in exponential_logit]

        r = random.random()
        cumulative = 0

        for index, probability in enumerate(probabilities):
            cumulative += probability
            if r <= cumulative:
                return index

        return len(probabilities) - 1

    def expect_close_bracket(  # FORCES TOKEN TO BE ']' -> IS THIS RIGHT???!!!
        self,
        logits: List[float],
        _generated_tokens: str,
        _functions: List[Function]
    ) -> None:
        sorted_logits_ids: List[int] = [
            index
            for index, _ in sorted(
                enumerate(logits),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        for id in sorted_logits_ids:
            text: str = self.llm.decode(id)
            if text != ']':
                logits[id] = float('-inf')

        # print(f"expect_name: {[self.llm.decode([i]) for i, logit in enumerate(logits) if logit != float('-inf')]}")

    @staticmethod
    def is_valid(current: str, next: List[str]) -> bool:
        for option in next:
            if len(current) <= len(option):
                match: int = 0
                for i in range(len(current)):
                    if current[i] == option[i]:
                        match += 1
                    else:
                        break

                if match == len(current):
                    return True

        return False

    @staticmethod
    def generate_parameters_regex(parameters: List[str]) -> List[re.Pattern]:  # "-?\\d+\\.\\d+"
        number_regex = re.compile(r"^-?\d+\.\d+$")
        string_regex = re.compile(r'^"([^"\\]|\\.)+"$')
        regex_list: List[re.Pattern] = []
        # parameters ["a", "number", "b", "number"]
        for _, value in enumerate(parameters):
            regex_list.append(
                number_regex if value == "number" else string_regex
            )

        return regex_list
        # return (",").join(regex_list)

    def found_parameter(self, generated_parameters_tokens: str) -> bool:
        # parameter: str = self.current_parameters[self.parameters_index]

        # if self.parameters_index % 2 != 0:
        #     pattern = TokenParser.generate_parameters_regex(
        #         ["", parameter]
        #     )[3:]
        # else:
        #     pattern = TokenParser.generate_parameters_regex([parameter])

        # if self.parameters_index != len(self.current_parameters) - 1:
        #     pattern += r',$'
        # else:
        #     pattern += r'$'
        pattern = self.parameters_regex[self.parameters_index]
        # print(f"pattern for {self.current_parameters[self.parameters_index]} is {pattern}")

        result = (
            False
            if re.match(pattern, generated_parameters_tokens) is None
            else True
        ) 
        print(f"{pattern}.match({generated_parameters_tokens}):{result}")
        return result

    def update_current_state(self, generated_tokens: str, functions: List[Function]) -> bool:
        basic_pattern: str = r'"([^"\\]|\\.)+"'
        if self.current_state == State.DONE.value:
            return True
        # if (
        #     self.current_state == State.EXPECT_PARAMETERS.value and
        #     re.match(
        #         r'(?:' + self.parameters_regex + r')$',
        #         generated_tokens
        #     ) is not None
        # ):
        #     self.current_state = State.DONE.value
        #     return True
        # if (
        #     re.match(
        #         basic_pattern + r',(?:' + self.parameters_regex + r')\]$',
        #         generated_tokens
        #     ) is not None
        # ):
        #     self.current_state = State.DONE.value
        #     return True
        # elif re.match(
        #     basic_pattern + r',(?:' + self.parameters_regex + r')',
        #     generated_tokens  # DOES $ NEED TO BE AT END?????
        # ) is not None:
        #     # print(f"self.parameters_regex in rematch 3 is {self.parameters_regex}")
        #     self.current_state = State.EXPECT_CLOSE_BRACKET.value
        # elif re.match(
        #     basic_pattern + r',(?:(?:' + self.parameters_regex + r'))?',
        #     generated_tokens
        # ) is not None:
        #     # print(f"self.parameters_regex in rematch 2 is {self.parameters_regex}")
        #     self.current_state = State.EXPECT_PARAMETERS.value
        if (
            self.current_state == State.EXPECT_NAME.value and
            re.match(basic_pattern, generated_tokens) is not None
        ): # if f_name is not None then move to expect comma & save f_name & generate full regex
            # self.current_state = State.EXPECT_COMMA.value
            self.update_values(generated_tokens, functions)
            self.current_state = State.EXPECT_PARAMETERS.value
            return True
        # elif generated_tokens.startswith("["):
        #     self.current_state = State.EXPECT_NAME.value

        return False
