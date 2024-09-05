from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from llm import llm

# 假设有一个外部函数用来查询天气，这里我们模拟它的行为
def get_weather(city: str):
    # 模拟返回城市的天气信息
    return f"The weather in {city} is sunny with a temperature of 25°C."

# 定义工具的输入模式
class WeatherToolInput(BaseModel):
    city: str = Field(..., description="The name of the city to get the weather for.")

# 自定义一个工具，用来查询天气
class WeatherTool(BaseTool):
    name = "WeatherTool"
    description = "A tool that provides the current weather for a given city."
    args_schema = WeatherToolInput  # 定义工具输入模式

    def _run(self, city: str, **kwargs):
        # 调用查询天气的逻辑
        return get_weather(city)

# 创建一个 LLM 实例
# llm = llm  # 假设你已经定义了一个 llm 实例，比如 ChatOpenAI

# 创建一个工具列表
tools = [WeatherTool()]

# 定义系统和用户的 prompt，用来引导 agent 生成正确的输出
system_prompt = '''You are a helpful assistant with access to the following tool:

{tools}

Use a JSON blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Begin! Always respond with a valid JSON blob.'''

human_prompt = '''{input}

{agent_scratchpad}

(Always respond in a JSON blob format as described above)'''

# 创建 ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human_prompt),
    ]
)

# 创建一个结构化的聊天 agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# 创建 agent 执行器来运行 agent
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 测试 agent，输入一个问题来触发工具的调用
response = agent_executor.invoke({"input": "What is the weather in Shanghai?"})
print(response)
