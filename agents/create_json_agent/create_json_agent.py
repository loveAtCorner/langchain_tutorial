import os
import yaml
from langchain_community.agent_toolkits.json.base import create_json_agent
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_community.tools.json.tool import JsonSpec
from llm import llm

with open("openapi.yaml", encoding="utf-8") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=llm, toolkit=json_toolkit, verbose=True
)

json_agent_executor.invoke(
    "What are the required parameters in the request body to the /completions endpoint?"
)

