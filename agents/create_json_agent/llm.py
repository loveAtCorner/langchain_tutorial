import sys
import os

# 获取上级目录路径并添加到 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# 获取上两级目录路径并添加到 sys.path
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(grandparent_dir)

from config import MODEL_CONFIGS  # 从上级目录的 config.py 导入 MODEL_CONFIGS
from langchain_openai import ChatOpenAI

# 选择要使用的模型
model_key = "model_3"  # 可以根据需要选择模型的键值

# 从配置文件中读取对应的模型配置
model_config = MODEL_CONFIGS.get(model_key)

# 检查是否成功读取配置
if model_config:
    llm = ChatOpenAI(
        model_name=model_config["model_name"],
        openai_api_base=model_config["openai_api_base"],
        openai_api_key=model_config["openai_api_key"]
    )
else:
    raise ValueError(f"未找到模型 {model_key} 的配置。")

if __name__ == "__main__":
    # 输入提示（Prompt）
    prompt = """
    你是一位经验丰富的电信运营商客户服务人员，用户的问题是：
    "近期我发现自己所在区域的手机信号不稳定，通话经常中断，请帮我解决这个问题。"
    """

    # 使用invoke方法调用ChatOpenAI模型生成响应
    response = llm.invoke(prompt)

    # 输出模型的响应
    print("模型的响应: ", response)
