from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from llm import llm

template = '''
请你扮演一为电信运营商客户服务人员，通过我提供给你的文本，简洁明了的回答用户的问题。

文本:
您好！我是贵公司的忠实用户，张三（手机号码：138xxxxxx88），长期以来一直信赖并享受着贵公司提供的通信服务。然而，近期我遇到了一系列服务问题，
对我的日常生活及工作造成了不小的困扰，特此向您反映并寻求紧急帮助。

近期，我发现自己所在区域的手机信号出现了频繁的波动与不稳定现象，导致通话时常中断，信息发送延迟，严重影响了我的日常通讯需求。
同时，家中的宽带服务也多次无预警地中断，给我的在线办公、学习以及家庭娱乐带来了极大的不便。
更为遗憾的是，在尝试通过客服渠道寻求帮助时，遇到了响应迟缓的情况，这让我感到十分焦急与无助。
问题:{question}
'''

print(PromptTemplate.from_template(template))

# 填充模型名
llm = llm
output_parser = StrOutputParser()
chain = (
        {"question": RunnablePassthrough()}
        | PromptTemplate.from_template(template)
        | llm
        | output_parser)

print(chain.invoke("请提供帮助"))
