from langchain_core.tools import tool
from typing import Annotated, List
from langchain.pydantic_v1 import BaseModel, Field


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiply2(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Let's inspect some of the attributes associated with the tool.
print(multiply.name)
print(multiply.description)
print(multiply.args)

print(multiply2.name)
print(multiply2.description)
print(multiply2.args)
print(multiply2.return_direct)