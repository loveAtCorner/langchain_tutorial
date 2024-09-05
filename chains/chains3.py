from typing import List
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from llm import llm

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")  # 笑话设置的部分
    punchline: str = Field(description="answer to resolve the joke")  # 笑话的冷笑话部分

    # 对笑话进行逻辑验证，保证setup的部分以？结尾
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field

# 初始化ChatOpenAI实例
llm = llm
# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

print(prompt)

chain = prompt | llm | parser

print(chain.invoke({"query": joke_query}))
