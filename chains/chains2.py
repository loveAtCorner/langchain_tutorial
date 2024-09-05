from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import llm

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class technical documentation writer."),
    ("user", "{input}")
])

# 填充模型名
llm = llm

# 定义输出解析器
output_parser = StrOutputParser()

# 创建链条
chain = prompt | llm | output_parser

# 生成 prompt 的值
prompt_value = prompt.invoke({"input": "how can langsmith help with testing?"})
print(prompt_value)

# 通过链条获取模型的响应
response = chain.invoke(prompt_value)

# 打印模型响应
print(response)
