from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from llm import llm

with open("政府工作报告.txt", encoding="utf-8") as f:
    report_2023 = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(report_2023)
docs = [Document(page_content=t) for t in texts]
prompt_template = """对下面的文字做精简的摘要:

    {text}

    """

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)
summ = chain({"input_documents": docs}, return_only_outputs=True)
print(summ['output_text'])