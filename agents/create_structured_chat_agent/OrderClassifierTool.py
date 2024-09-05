from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from llm import llm  # 假设你已经定义了使用 OpenAI 的模型

"""
### 代码说明：
1. **提示词工程**：`_classify_ticket_with_llm` 方法通过大语言模型的提示词，将工单内容根据语义分类为“资费”、“套餐”或“服务”。
2. **分类逻辑**：工单分类结果只能是“资费”、“套餐”或“服务”，如果大语言模型的输出不是这三类中的一类，则返回“未知分类”。
3. **JSON 格式输出**：分类结果会以 JSON 格式输出，并包含分类标签和工单内容。

这个工具可以根据用户提供的工单内容自动进行分类，并在需要时处理解析错误。
"""

# 定义工单分类工具的输入数据结构
class TicketClassificationToolInput(BaseModel):
    sentence: str = Field(..., description="需要分类的工单内容")

# 定义工单分类工具
class TicketClassificationTool(BaseTool):
    name = "TicketClassification"
    description = "根据工单的语义，将其分类为资费、套餐或服务"

    args_schema = TicketClassificationToolInput  # 使用 Pydantic 定义输入参数

    def _classify_ticket_with_llm(self, sentence: str) -> str:
        """通过大语言模型进行工单分类"""
        prompt = f"根据下面的句子内容，将其分类为'资费'、'套餐'或'服务'。只返回一个分类标签。\n句子: {sentence}"
        response = llm.invoke(prompt)  # 使用 LLM 预测并分类
        category = response.content.strip()  # 提取返回的分类标签
        return category

    def _run(self, sentence: str):
        """工单分类"""
        category = self._classify_ticket_with_llm(sentence)
        
        # 如果没有匹配的分类，返回未知分类
        if category not in ["资费", "套餐", "服务"]:
            return "未知分类"
        
        # 返回分类结果
        return {
            "action": "Final Answer",
            "action_input": {
                "分类": category
            }
        }

# 创建工具列表
tools = [TicketClassificationTool()]

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
user_input = "被投诉工号:dndGDB；营业厅名称:长治潞城区店上潞卓农村泛渠道；13712347701用户反映没有办理套餐升档享流量及数字权益 2023-07-3015:29:59，工号： dndGDB，要求取消处理并改回原套餐。 受理节点：移动业务→业务营销→本省业务营销→社会渠道→办理规范→本省问题→全局流转 请尽快处理，一个小时后回访客户，如未处理将形成工单。"
response = agent_executor.invoke({"input": user_input})
print(response)
