from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from llm import llm

# 初始化ChatOpenAI实例
llm = llm

# 定义样例数据
examples = [
    {"UserInput": "被投诉工号:abc123；营业厅名称:广州天河区店；13876543210用户反映本月资费异常增加，要求核实账单，2023-08-0110:15:23。工号： abc123，要求在24小时内解决，如未处理将投诉到上级部门。", "IntentType": "资费"},
    {"UserInput": "被投诉工号:hij234；营业厅名称:北京海淀区店；13123456789用户反映套餐升档后流量未增加，2023-08-0412:45:59。工号： hij234，要求重新核查套餐内容并恢复原套餐。", "IntentType": "套餐"},
    {"UserInput": "被投诉工号:stu012；营业厅名称:杭州西湖区店；13456789012用户反映营业厅服务态度差，2023-08-0716:15:08。工号： stu012，要求道歉并给予服务改进方案。", "IntentType": "服务"},
]

# 定义example_prompt模板
example_prompt = PromptTemplate(
    input_variables=["UserInput", "IntentType"],
    template="UserInput: {UserInput}\nIntentType: {IntentType}",
)

# 使用 LengthBasedExampleSelector 来选择合适长度的样例
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=500
)

# 定义响应模式和解析器
response_schemas = [
    ResponseSchema(name="UserInput", description="the description of work orders"),
    ResponseSchema(name="IntentType", description="the classification type"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# 构建 FewShotPromptTemplate
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="你是一名NLP算法工程师，现在你需要执行一个意图识别分类的任务：理解用户输入并将其匹配到对应的意图。为了得到符合要求的答案，请按照下面的过程，一步步思考并得出回答：1、你需要先了解意图信息一共有如下三个类别：资费、套餐、服务。这三个类别的具体含义如下：资费：指用户对当前资费或账单有疑问或异常。例如，用户反映资费异常增加、被收取了不明附加费用，或者套餐中的某些资费内容未按预期提供等。该类意图主要涉及用户对资费的核实、调整或争议的解决。套餐：指用户对当前使用的套餐内容或变更有疑问或需求。例如，用户反映套餐升档后服务未按预期提供，套餐变更过程中费用有误，或者对套餐的服务内容存在争议等。该类意图涉及套餐的核实、调整、变更或恢复等需求。服务：指用户对营业厅或客服的服务质量、技术支持等有疑问或不满。例如，用户投诉服务态度差、网络故障未及时处理、客服电话无人接听等。该类意图涉及服务质量的反馈、技术支持的请求或客服响应速度的改善等。其他：用户描述不符合上述任何意图，则分类到该意图。2、你要做的是只需要理解用户输入内容是想要干什么，然后对应到相应的意图类别即可。",
    suffix="UserInput: {noun}\nIntentType:",
    input_variables=["noun"],
    partial_variables={"format_instructions": format_instructions},
)

# 构建最终的提示词模板，包含样例并严格规定输出格式
final_prompt = PromptTemplate(
    template="{dynamic_prompt}\n{format_instructions}\nQuestion: {question}",
    input_variables=["dynamic_prompt", "question"],
    partial_variables={"format_instructions": format_instructions},
)
print(final_prompt)

# 生成并输出结果
chain = final_prompt | llm | output_parser
result = chain.invoke({
    "dynamic_prompt": dynamic_prompt.invoke({"noun": "被投诉工号:dndGDB；营业厅名称:长治潞城区店上潞卓农村泛渠道；13712347701用户反映没有办理套餐升档享流量及数字权益 2023-07-3015:29:59，工号： dndGDB，要求取消处理并改回原套餐。 受理节点：移动业务→业务营销→本省业务营销→社会渠道→办理规范→本省问题→全局流转 请尽快处理，一个小时后回访客户，如未处理将形成工单。"}).text,
    "question": "请识别这句话所归属的意图类别"
})

print(result)