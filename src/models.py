from pydantic import BaseModel, model_validator, ValidationError
from pydantic_core import PydanticCustomError
from typing import List, Dict, Any, Self
from dataclasses import dataclass


class Prompt(BaseModel):
    prompt: str

    @model_validator(mode='after')
    def check_empty_prompt(self) -> Self:
        if self.prompt == "":
            raise ValidationError.from_exception_data(
                title="Prompt",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_empty_prompt",
                        "Prompt is empty"
                    ),
                    "loc": ("prompt",),
                    "input": {
                        "prompt": self.prompt
                    }
                }]
            )
        return self


@dataclass
class PromptResponse:
    prompt: str
    name: str
    parameters: Dict[str, Any]


class Type(BaseModel):
    type: str

    @model_validator(mode='after')
    def check_empty_type(self) -> Self:
        if self.type == "":
            raise ValidationError.from_exception_data(
                title="Type",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_empty_type",
                        "Type is empty"
                    ),
                    "loc": ("type",),
                    "input": {
                        "type": self.type
                    }
                }]
            )
        return self

    @model_validator(mode='after')
    def check_wrong_type(self) -> Self:
        if self.type not in ["number", "string"]:
            raise ValidationError.from_exception_data(
                title="Type",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_wrong_type",
                        "Wrong type"
                    ),
                    "loc": ("type",),
                    "input": {
                        "type": self.type
                    }
                }]
            )
        return self


class Parameter(BaseModel):
    name: str
    type: Type

    @model_validator(mode='after')
    def check_empty_param_name(self) -> Self:
        if self.name == "":
            raise ValidationError.from_exception_data(
                title="Name",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_empty_param_name",
                        "Parameter name is empty"
                    ),
                    "loc": ("name",),
                    "input": {
                        "name": self.name
                    }
                }]
            )
        return self


class Function(BaseModel):
    name: str
    description: str
    parameters: List[Parameter]
    returns: Type

    @model_validator(mode='after')
    def check_empty_func_name(self) -> Self:
        if self.name == "":
            raise ValidationError.from_exception_data(
                title="Name",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_empty_func_name",
                        "Function name is empty"
                    ),
                    "loc": ("name",),
                    "input": {
                        "name": self.name
                    }
                }]
            )
        return self

    @model_validator(mode='after')
    def check_empty_func_desc(self) -> Self:
        if self.description == "":
            raise ValidationError.from_exception_data(
                title="Description",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_empty_func_desc",
                        "Function description is empty"
                    ),
                    "loc": ("description",),
                    "input": {
                        "description": self.description
                    }
                }]
            )
        return self
