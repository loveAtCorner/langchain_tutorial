from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import requests

from llm import llm  # 假设你已经定义了使用 OpenAI 的模型

"""
### 代码说明：
1. **提示词工程**：`_extract_city_with_llm` 方法通过提示词与大语言模型（如 GPT）进行交互，从用户的输入中提取城市名称。提示词要求模型只返回城市名称，并且如果没有找到城市名称则返回“未知城市”。
2. **API 调用**：一旦提取到城市名称，`WeatherTool` 会使用 API 来获取该城市的天气信息。
3. **集成流程**：用户输入完整的句子，`WeatherTool` 会先通过 LLM 提取城市名称，再调用天气查询功能。

这种方法利用大语言模型的自然语言理解能力，更加智能地从用户输入中提取城市名称，不再依赖简单的正则匹配。
"""

# 替换为你的API密钥
API_KEY = 'e942c23175702a30a5533a249215b4f2'

# 定义工具的输入数据结构
class WeatherToolInput(BaseModel):
    sentence: str = Field(..., description="用户输入的包含城市名称的句子。")

# 定义一个优化后的 WeatherTool，利用提示词工程从输入中提取城市名称
class WeatherTool(BaseTool):
    name = "WeatherQuery"
    description = "从用户的句子中提取城市名称并查询该城市的天气。"
    
    args_schema = WeatherToolInput  # 使用 Pydantic 定义输入参数

    def _extract_city_with_llm(self, sentence: str) -> str:
        """通过大语言模型提取城市名称"""
        prompt = f"从下面的句子中提取城市名称。只返回城市名称。如果句子中没有城市名称，返回'未知城市'。\n句子: {sentence}"
        response = llm.invoke(prompt)  # 使用 LLM 来预测并提取城市名称
        city_name = response.content.strip()
        return city_name 

    def _run(self, sentence: str):
        """提取城市名称并查询天气"""
        city = self._extract_city_with_llm(sentence)
        
        if city == "未知城市":
            return "无法从输入中提取城市名称。"
        
        # 使用实际的天气查询 API
        url = 'http://apis.juhe.cn/simpleWeather/query'
        params = {
            'city': city,
            'key': API_KEY
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['error_code'] == 0:
                result = data['result']
                realtime = result['realtime']
                future = result['future']

                # 格式化当前天气和未来的天气预报
                current_weather = (
                    f"当前天气: {realtime['info']}, 温度: {realtime['temperature']}°C, "
                    f"湿度: {realtime['humidity']}%"
                )
                future_forecast = "\n未来天气预报:\n"
                for day in future:
                    future_forecast += f"{day['date']}: {day['weather']}, 温度: {day['temperature']}\n"

                # 生成符合 JSON 格式的输出，确保可以被解析
                return {
                    "action": "Final Answer",
                    "action_input": {
                        "城市": city,
                        "当前天气": current_weather,
                        "未来天气预报": future_forecast
                    }
                }
            else:
                return f"请求失败: {data['reason']}"
        else:
            return f"HTTP请求失败: {response.status_code}"

# 创建工具列表
tools = [WeatherTool()]

# 定义系统提示和用户提示的结构化对话（中文版本）
system_prompt = '''尽可能帮助并准确地回应人类。你可以使用以下工具：

{tools}

使用一个 JSON 数据块来指定工具，通过提供 "action" 键（工具名称）和 "action_input" 键（工具输入）。

有效的 "action" 值为："Final Answer" 或 {tool_names}

每个 $JSON_BLOB 仅包含一个操作，如下所示：

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

开始吧！记得一定要用有效的 JSON 数据块来回复。'''

human_prompt = '''{input}

{agent_scratchpad}

（记得无论如何都要使用 JSON 数据块回复）'''

# 创建提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human_prompt),
    ]
)

# 使用 OpenAI 的语言模型
# llm = ChatOpenAI(temperature=0)  # 确保在你的环境中正确定义

# 创建结构化聊天代理
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# 创建代理执行器，增加 handle_parsing_errors 以处理解析错误
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# 测试代理执行器，输入一个问题来触发工具调用
user_input = "请问上海的天气怎么样？"
response = agent_executor.invoke({"input": user_input})
print(response)
