from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
import asyncio

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(
        self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> int:
        """Use the tool."""
        return a * b

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> int:
        """Use the tool asynchronously."""
        return self._run(a, b, run_manager=run_manager.get_sync() if run_manager else None)


def main():
    multiply = CustomCalculatorTool()
    print(multiply.name)
    print(multiply.description)
    print(multiply.args_schema)
    print(multiply.return_direct)

    # 同步调用
    print(multiply.invoke({"a": 2, "b": 3}))

    # 异步调用
    asyncio.run(run_async(multiply))


async def run_async(multiply):
    # 异步调用需要放在异步函数内
    result = await multiply.ainvoke({"a": 2, "b": 3})
    print(result)


if __name__ == "__main__":
    main()
