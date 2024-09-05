from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from llm import llm 

# 我们可以首先将其提取为字符串。
memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")

print(memory.load_memory_variables({}))

# 我们还可以将历史记录作为消息列表获取
memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")

print(memory.load_memory_variables({}))

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

print(conversation.predict(input="Hi there!"))
print(conversation.predict(input="I'm doing well! Just having a conversation with an AI."))
print(conversation.predict(input="Tell me about yourself."))