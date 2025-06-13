from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class FunctionParameterProperty(BaseModel):
    """Describes a single property within a function's parameters schema."""
    type: str = Field(..., description="Type of the parameter (e.g., 'string', 'integer', 'boolean', 'object', 'array').")
    description: Optional[str] = Field(None, description="Description of the parameter.")
    # For more complex schemas, one might add 'items' for arrays, 'properties' for objects, 'enum' for specific values.
    # For now, keeping it simple for typical flat structures used in extraction.

class FunctionParameters(BaseModel):
    """Defines the schema for parameters a function accepts, mirroring OpenAPI schema subset."""
    type: str = Field(default="object", description="The overall type of the parameters object, typically 'object'.")
    properties: Dict[str, FunctionParameterProperty] = Field(..., description="Dictionary of parameter names to their schema definitions.")
    required: Optional[List[str]] = Field(None, description="List of required parameter names.")

class FunctionDeclaration(BaseModel):
    """
    Describes a function (tool) that the LLM can call.
    """
    name: str = Field(..., description="The name of the function to be called.")
    description: str = Field(..., description="A description of what the function does and when to use it.")
    parameters: FunctionParameters = Field(..., description="The schema defining the parameters the function accepts.")

class Tool(BaseModel):
    """
    A tool that can be provided to the LLM, consisting of one or more function declarations.
    Corresponds to Gemini's `Tool` which contains `function_declarations`.
    """
    function_declarations: List[FunctionDeclaration]

class FunctionCall(BaseModel):
    """
    Represents a function call requested by the LLM.
    """
    name: str = Field(..., description="Name of the function the LLM wants to call.")
    # Arguments are expected to be a dict, but can be complex. Using Any for now.
    # In practice, this would be Dict[str, Any] where Any can be primitive types, lists, or nested dicts.
    arguments: Dict[str, Any] = Field(..., description="A dictionary of arguments to pass to the function, conforming to the function's parameter schema.")

class LLMResponse(BaseModel):
    """
    Represents a response from an LLM, which can be either a direct text response
    or one or more function calls requested by the LLM.
    """
    text: Optional[str] = Field(None, description="Direct text response from the LLM.")
    function_calls: Optional[List[FunctionCall]] = Field(None, description="List of function calls requested by the LLM.")

    # Add a validator to ensure that either text or function_calls is provided, but not both (or neither if that's not allowed).
    # For now, this is a simple container. Specific clients will ensure correct population.
