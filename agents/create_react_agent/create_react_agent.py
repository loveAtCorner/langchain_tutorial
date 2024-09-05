from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from llm import llm

# 假设有一个查询天气的函数
def get_weather(city: str):
    return f"The weather in {city} is sunny with 25°C."

# 定义工具输入的模式
class WeatherToolInput(BaseModel):
    city: str = Field(..., description="The name of the city to get the weather for.")

# 自定义工具，用于查询天气
class WeatherTool(BaseTool):
    name = "WeatherTool"
    description = "Provides the current weather for a city."
    args_schema = WeatherToolInput

    def _run(self, city: str):
        return get_weather(city)

# 创建语言模型实例（假设已定义或导入一个 LLM 实例）
# llm = BaseLanguageModel()  # 示例 LLM，需要用你自己的模型替换

# 创建工具列表
tools = [WeatherTool()]

# 定义 ReAct 提示模板
template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# 创建 ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 创建 AgentExecutor 执行器
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 测试 agent，输入问题以触发工具调用
response = agent_executor.invoke({"input": "What is the weather in New York?"})
print(response)
