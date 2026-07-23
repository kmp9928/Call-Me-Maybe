from typing import List, Dict, Any, Self
from pydantic import BaseModel, model_validator, ValidationError
from pydantic_core import PydanticCustomError


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


class Returns(BaseModel):
    type: str

    @model_validator(mode='after')
    def check_wrong_type(self) -> Self:
        if self.type not in ["number", "integer", "string", "boolean"]:
            raise ValidationError.from_exception_data(
                title="Returns",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_wrong_type",
                        "Wrong returns type"
                    ),
                    "loc": ("type",),
                    "input": {
                        "returns": self.type
                    }
                }]
            )
        return self


class Parameter(BaseModel):
    name: str
    type: str

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

    @model_validator(mode='after')
    def check_wrong_type(self) -> Self:
        if self.type not in ["number", "integer", "string", "boolean"]:
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


class Function(BaseModel):
    name: str
    description: str
    parameters: List[Parameter]
    returns: Returns

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


class PromptResponse(BaseModel):
    prompt: str
    function: Function
    parameters: Dict[str, Any]

    @model_validator(mode='after')
    def check_parameters(self) -> Self:
        original_parameters = [p.name for p in self.function.parameters]
        output_parameters = [p for p in self.parameters.keys()]
        if original_parameters != output_parameters:
            raise ValidationError.from_exception_data(
                title="Parameters",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_parameters",
                        "Output parameters don't match original ones"
                    ),
                    "loc": ("name",),
                    "input": {
                        "parameters": output_parameters
                    }
                }]
            )
        return self
