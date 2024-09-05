from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate, load_prompt
from llm import llm

# 初始化ChatOpenAI实例
llm = llm

# 响应模式指导LLM的输出解析格式，传入StructuredOutputParser
response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="source", description="source used to answer the user's question, should be a website."),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)
print(prompt)

chain = prompt | llm | output_parser
print(chain.invoke({"question": "what's the capital of france?"}))
