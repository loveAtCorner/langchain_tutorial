from langchain_core.tools import StructuredTool

"""
create tools from functions
"""

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)

def main():
    # 同步调用
    print(calculator.invoke({"a": 2, "b": 3}))

    # 异步调用
    import asyncio
    asyncio.run(run_async())

async def run_async():
    # 异步调用需要放在异步函数内
    print(await calculator.ainvoke({"a": 2, "b": 5}))

if __name__ == "__main__":
    main()
