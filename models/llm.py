from langchain_openai import ChatOpenAI

# 初始化ChatOpenAI实例
llm = ChatOpenAI()

# 输入提示（Prompt）
prompt = """
你是一位经验丰富的电信运营商客户服务人员，用户的问题是：
"近期我发现自己所在区域的手机信号不稳定，通话经常中断，请帮我解决这个问题。"
"""

# 使用invoke方法调用ChatOpenAI模型生成响应
response = llm.invoke(prompt)

# 输出模型的响应
print("模型的响应: ", response)

