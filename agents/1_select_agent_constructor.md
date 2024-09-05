# agent constructor 

接下来我们逐一介绍各个 agent constructor 的使用场景
- create_react_agent
    - 适用于需要多步推理的场景，尤其是当模型需要在解决问题过程中反复思考和调用不同的工具时。这种代理能够通过反复思考（thought）和行动（action）来灵活处理复杂任务。
- create_json_agent
    - 是一个适用于与大型 JSON/dict 对象进行交互的代理。当您想要回答关于一个超出 LLM 上下文窗口大小的 JSON 数据块的问题时，这将非常有用。该代理能够迭代地探索数据块，找到回答用户问题所需的信息。
- create_structured_chat_agent
    - 适用于需要多个工具并处理复杂输入的场景，特别是在需要精确调用工具并生成结构化输出时。

## create_react_agent

```python
from __future__ import annotations

from typing import List, Optional, Sequence, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.tools.render import ToolsRenderer, render_text_description

from langchain.agents import AgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser


def create_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    output_parser: Optional[AgentOutputParser] = None,
    tools_renderer: ToolsRenderer = render_text_description,
    *,
    stop_sequence: Union[bool, List[str]] = True,
) -> Runnable:
    """Create an agent that uses ReAct prompting.

    Based on paper "ReAct: Synergizing Reasoning and Acting in Language Models"
    (https://arxiv.org/abs/2210.03629)

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more.
        output_parser: AgentOutputParser for parse the LLM output.
        tools_renderer: This controls how the tools are converted into a string and
            then passed into the LLM. Default is `render_text_description`.
        stop_sequence: bool or list of str.
            If True, adds a stop token of "Observation:" to avoid hallucinates.
            If False, does not add a stop token.
            If a list of str, uses the provided list as the stop tokens.

            Default is True. You may to set this to False if the LLM you are using
            does not support stop sequences.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Examples:

        .. code-block:: python

            from langchain import hub
            from langchain_community.llms import OpenAI
            from langchain.agents import AgentExecutor, create_react_agent

            prompt = hub.pull("hwchase17/react")
            model = OpenAI()
            tools = ...

            agent = create_react_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

            # Use with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    # Notice that chat_history is a string
                    # since this prompt is aimed at LLMs, not chat models
                    "chat_history": "Human: My name is Bob\\nAI: Hello Bob!",
                }
            )

    Prompt:

        The prompt must have input keys:
            * `tools`: contains descriptions and arguments for each tool.
            * `tool_names`: contains all tool names.
            * `agent_scratchpad`: contains previous agent actions and tool outputs as a string.

        Here's an example:

        .. code-block:: python

            from langchain_core.prompts import PromptTemplate

            template = '''Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}'''

            prompt = PromptTemplate.from_template(template)
    """  # noqa: E501
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    if stop_sequence:
        stop = ["\nObservation"] if stop_sequence is True else stop_sequence
        llm_with_stop = llm.bind(stop=stop)
    else:
        llm_with_stop = llm
    output_parser = output_parser or ReActSingleInputOutputParser()
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | output_parser
    )
    return agent
```


`create_react_agent` 函数的主要功能是创建一个基于 ReAct (Reasoning and Acting) 的智能体，它使用一个大语言模型（LLM）和多个工具来处理用户的输入并做出相应的决策。

### 该函数的功能概述：
1. **输入参数**：
   - `llm`: 使用的大语言模型，用于生成回答。
   - `tools`: 该智能体可以访问的工具列表，用于辅助完成特定任务。
   - `prompt`: 提示模板，控制智能体的思维过程和行为逻辑。
   - `output_parser`: 可选的输出解析器，用于解析 LLM 的输出。
   - `tools_renderer`: 控制工具如何被转化为字符串传递给 LLM。
   - `stop_sequence`: 用于控制 LLM 生成回答的终止条件，防止生成虚假的观察结果。

2. **功能**：
   - 根据 ReAct 框架创建一个智能体，能够在回答问题时调用工具完成特定任务，并生成回答。
   - 构建提示模板，确保 LLM 按照正确的格式进行思维、执行动作并返回观察结果。
   - 根据用户输入和上下文生成交互式输出，智能体可以根据需求重复执行行动，直到得到最终答案。

3. **返回值**：返回一个可运行的 `Runnable` 序列，代表 ReAct agent，它接受输入并产生最终输出。

4. **典型应用**：该智能体可以应用于复杂的任务，例如问题解答和工具调用，确保通过多步推理得出最终答案。



## create_json_agent

```python
"""Json agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel

from langchain_community.agent_toolkits.json.prompt import JSON_PREFIX, JSON_SUFFIX
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit

if TYPE_CHECKING:
    from langchain.agents.agent import AgentExecutor


def create_json_agent(
    llm: BaseLanguageModel,
    toolkit: JsonToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = JSON_PREFIX,
    suffix: str = JSON_SUFFIX,
    format_instructions: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a json agent from an LLM and tools.

    Args:
        llm: The language model to use.
        toolkit: The toolkit to use.
        callback_manager: The callback manager to use. Default is None.
        prefix: The prefix to use. Default is JSON_PREFIX.
        suffix: The suffix to use. Default is JSON_SUFFIX.
        format_instructions: The format instructions to use. Default is None.
        input_variables: The input variables to use. Default is None.
        verbose: Whether to print verbose output. Default is False.
        agent_executor_kwargs: Optional additional arguments for the agent executor.
        kwargs: Additional arguments for the agent.

    Returns:
        The agent executor.
    """
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.chains.llm import LLMChain

    tools = toolkit.get_tools()
    prompt_params = (
        {"format_instructions": format_instructions}
        if format_instructions is not None
        else {}
    )
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
        **prompt_params,
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )
```


`create_json_agent` 函数的主要功能是基于大语言模型（LLM）和工具集创建一个 JSON 解析和处理的智能体，该智能体能够使用指定工具执行任务并输出结果。

### 该函数的功能概述：
1. **输入参数**：
   - `llm`: 使用的语言模型，用于生成答案或执行任务。
   - `toolkit`: 包含工具的工具集，用于辅助智能体处理任务。
   - `callback_manager`: 可选的回调管理器，用于管理执行过程中的回调操作。
   - `prefix` 和 `suffix`: 提示前缀和后缀，定义智能体如何与工具进行交互。
   - `format_instructions`: 可选的格式化指令，提供给语言模型处理的格式要求。
   - `input_variables`: 可选的输入变量列表，定义提示中需要的输入。
   - `verbose`: 控制是否打印详细信息的标志。
   - `agent_executor_kwargs`: 传递给智能体执行器的额外参数。

2. **功能**：
   - 创建一个基于 ZeroShotAgent 的 JSON 解析智能体，结合 LLM 和工具来完成指定任务。
   - 生成提示模板，指导智能体如何使用工具进行任务的思考和执行。
   - 使用 `LLMChain` 构建 LLM 和提示的执行链，将工具结合到智能体的执行流中。
   - 返回一个 `AgentExecutor` 实例，用于调用智能体处理 JSON 数据。

3. **返回值**：返回一个 `AgentExecutor`，它能够运行智能体并处理输入。

### 典型应用：
该函数可以用于构建一个能够解析和处理 JSON 数据的智能体，通过语言模型与特定工具的协作，完成复杂的数据处理和推理任务。



## create_structured_chat_agent

```python
import re
from typing import Any, List, Optional, Sequence, Tuple, Union

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.tools.render import ToolsRenderer

from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.structured_chat.output_parser import (
    StructuredChatOutputParserWithRetries,
)
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.chains.llm import LLMChain
from langchain.tools.render import render_text_description_and_args

HUMAN_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"


@deprecated("0.1.0", alternative="create_structured_chat_agent", removal="1.0")
class StructuredChatAgent(Agent):
    """Structured Chat Agent."""

    output_parser: AgentOutputParser = Field(
        default_factory=StructuredChatOutputParserWithRetries
    )
    """Output parser for the agent."""

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        agent_scratchpad = super()._construct_scratchpad(intermediate_steps)
        if not isinstance(agent_scratchpad, str):
            raise ValueError("agent_scratchpad should be of type string.")
        if agent_scratchpad:
            return (
                f"This was your previous work "
                f"(but I haven't seen any of it! I only see what "
                f"you return as final answer):\n{agent_scratchpad}"
            )
        else:
            return agent_scratchpad

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        pass

    @classmethod
    def _get_default_output_parser(
        cls, llm: Optional[BaseLanguageModel] = None, **kwargs: Any
    ) -> AgentOutputParser:
        return StructuredChatOutputParserWithRetries.from_llm(llm=llm)

    @property
    def _stop(self) -> List[str]:
        return ["Observation:"]

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        human_message_template: str = HUMAN_MESSAGE_TEMPLATE,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        memory_prompts: Optional[List[BasePromptTemplate]] = None,
    ) -> BasePromptTemplate:
        tool_strings = []
        for tool in tools:
            args_schema = re.sub("}", "}}", re.sub("{", "{{", str(tool.args)))
            tool_strings.append(f"{tool.name}: {tool.description}, args: {args_schema}")
        formatted_tools = "\n".join(tool_strings)
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join([prefix, formatted_tools, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        _memory_prompts = memory_prompts or []
        messages = [
            SystemMessagePromptTemplate.from_template(template),
            *_memory_prompts,
            HumanMessagePromptTemplate.from_template(human_message_template),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)  # type: ignore[arg-type]

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        human_message_template: str = HUMAN_MESSAGE_TEMPLATE,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        memory_prompts: Optional[List[BasePromptTemplate]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            human_message_template=human_message_template,
            format_instructions=format_instructions,
            input_variables=input_variables,
            memory_prompts=memory_prompts,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser(llm=llm)
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    @property
    def _agent_type(self) -> str:
        raise ValueError


def create_structured_chat_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    tools_renderer: ToolsRenderer = render_text_description_and_args,
    *,
    stop_sequence: Union[bool, List[str]] = True,
) -> Runnable:
    """Create an agent aimed at supporting tools with multiple inputs.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more.
        stop_sequence: bool or list of str.
            If True, adds a stop token of "Observation:" to avoid hallucinates.
            If False, does not add a stop token.
            If a list of str, uses the provided list as the stop tokens.

            Default is True. You may to set this to False if the LLM you are using
            does not support stop sequences.
        tools_renderer: This controls how the tools are converted into a string and
            then passed into the LLM. Default is `render_text_description`.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Examples:

        .. code-block:: python

            from langchain import hub
            from langchain_community.chat_models import ChatOpenAI
            from langchain.agents import AgentExecutor, create_structured_chat_agent

            prompt = hub.pull("hwchase17/structured-chat-agent")
            model = ChatOpenAI()
            tools = ...

            agent = create_structured_chat_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

            # Using with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    "chat_history": [
                        HumanMessage(content="hi! my name is bob"),
                        AIMessage(content="Hello Bob! How can I assist you today?"),
                    ],
                }
            )

    Prompt:

        The prompt must have input keys:
            * `tools`: contains descriptions and arguments for each tool.
            * `tool_names`: contains all tool names.
            * `agent_scratchpad`: contains previous agent actions and tool outputs as a string.

        Here's an example:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            system = '''Respond to the human as helpfully and accurately as possible. You have access to the following tools:

            {tools}

            Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

            Valid "action" values: "Final Answer" or {tool_names}

            Provide only ONE action per $JSON_BLOB, as shown:

            ```
            {{
              "action": $TOOL_NAME,
              "action_input": $INPUT
            }}
            ```

            Follow this format:

            Question: input question to answer
            Thought: consider previous and subsequent steps
            Action:
            ```
            $JSON_BLOB
            ```
            Observation: action result
            ... (repeat Thought/Action/Observation N times)
            Thought: I know what to respond
            Action:
            ```
            {{
              "action": "Final Answer",
              "action_input": "Final response to human"
            }}

            Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'''

            human = '''{input}

            {agent_scratchpad}

            (reminder to respond in a JSON blob no matter what)'''

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", human),
                ]
            )
    """  # noqa: E501
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    if stop_sequence:
        stop = ["\nObservation"] if stop_sequence is True else stop_sequence
        llm_with_stop = llm.bind(stop=stop)
    else:
        llm_with_stop = llm

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | JSONAgentOutputParser()
    )
    return agent
```


`create_structured_chat_agent` 函数的主要功能是创建一个支持多输入工具的智能体，该智能体通过结构化对话与大语言模型（LLM）及多个工具进行交互，生成响应。

### 函数功能概述：
1. **输入参数**：
   - `llm`: 大语言模型，用于生成智能体的回答。
   - `tools`: 工具列表，智能体可以使用这些工具来执行特定任务。
   - `prompt`: 提示模板，定义智能体与用户和工具之间的交互方式。
   - `stop_sequence`: 控制生成响应的终止条件，避免虚假推理的停用标记。
   - `tools_renderer`: 控制工具如何被描述并传递给 LLM，默认为文本描述。

2. **功能**：
   - 创建一个 `Runnable` 序列，智能体能够根据输入数据调用多个工具，并执行相应操作。
   - 该智能体遵循结构化的对话模式，在每个步骤中根据输入决定使用哪个工具或生成响应。
   - 通过格式化提示模板，智能体可以考虑上下文信息，如之前的动作和工具的结果（称为“思考/行动/观察”循环），以执行复杂任务。

3. **返回值**：
   - 返回一个 `Runnable` 对象，代表智能体的执行流程，能够处理多步任务并返回最终结果。

### 典型应用：
该智能体可以应用于需要复杂推理的场景，例如当智能体需要反复调用多个工具来获得信息、执行操作，并在每个步骤生成部分回答，最终完成任务。




这三个函数 (`create_react_agent`, `create_json_agent`, `create_structured_chat_agent`) 都用于构建智能体，通过大语言模型（LLM）与工具集协作来处理任务，但它们各自的功能和应用场景有所不同。

### 1. **`create_react_agent`**:
   - **功能**：基于 ReAct 框架创建智能体。智能体能够根据用户输入进行推理（Reasoning）和行动（Acting），借助工具完成任务，并生成最终答案。
   - **应用场景**：适用于需要多步推理和行动的任务。智能体通过思考/行动/观察的循环，不断调用工具来处理复杂的用户请求。
   - **特点**：
     - 智能体根据预定义的提示模板操作。
     - 提供对工具的访问，智能体通过推理决定使用哪些工具。
     - 支持终止标记控制 LLM 的生成行为，避免生成无关内容。

### 2. **`create_json_agent`**:
   - **功能**：构建一个专门处理和解析 JSON 数据的智能体。智能体能够使用工具对 JSON 进行解析、处理，并返回结构化的结果。
   - **应用场景**：适用于需要处理 JSON 格式输入和输出的任务。该智能体可以解析 JSON 数据，并通过工具进一步处理数据。
   - **特点**：
     - 提供与 JSON 相关的工具集。
     - 使用语言模型与工具结合解析和处理 JSON。
     - 注重格式化输出，特别适用于以 JSON 形式返回结果的场景。

### 3. **`create_structured_chat_agent`**:
   - **功能**：创建一个支持多输入工具的结构化对话智能体。智能体能够在多个步骤中与工具交互，基于输入和中间步骤的结果生成响应。
   - **应用场景**：适用于需要通过多次交互处理复杂任务的场景，智能体能够在每一步调用不同工具，根据工具的结果生成新的操作。
   - **特点**：
     - 支持多工具和复杂的交互式任务。
     - 提供结构化的对话模式，智能体能够遵循“思考/行动/观察”模式反复调用工具。
     - 允许多种格式化指令和终止标记，以控制对话的顺序和工具的使用。

### **总结与区别**：
1. **功能差异**：
   - `create_react_agent`：专注于多步推理和行动，智能体根据工具的反馈推理和行动，解决复杂任务。
   - `create_json_agent`：专用于解析和处理 JSON 数据，智能体以 JSON 为核心进行操作。
   - `create_structured_chat_agent`：处理多输入工具的结构化对话，支持复杂任务的多次工具交互。

2. **应用场景差异**：
   - `create_react_agent`：用于需要智能体反复思考并通过工具解决问题的场景。
   - `create_json_agent`：用于解析、处理和返回 JSON 数据的任务。
   - `create_structured_chat_agent`：用于需要智能体通过多个工具和多个步骤来完成任务的结构化对话场景。

3. **相同点**：三者都依赖于语言模型和工具集，智能体能够根据用户输入结合工具执行任务，返回处理结果。