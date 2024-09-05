from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate)

from langchain.prompts.example_selector import LengthBasedExampleSelector

# 1. 定义example variables list
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]
# 2. 为examples定义example_prompt模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
# 3. 将example variables list和example_prompt输入到example_selector中
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25)
# 4. 定义few_shot_prompt模板（prefix + examples + suffix）
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",  # 前缀，定义task
    suffix="Input: {adjective}\nOutput:",  # 后缀
    input_variables=["adjective"],  # 后缀参数变量
)
print(dynamic_prompt.invoke({"adjective": "funny"}).text)
