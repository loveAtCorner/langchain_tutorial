from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 初始化ChatOpenAI实例
llm = ChatOpenAI()

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization?"),
]

response = llm.invoke(messages)

# 输出模型的响应
print("模型的响应: ", response)