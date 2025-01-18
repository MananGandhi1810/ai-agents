import docker
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv


client = docker.from_env()


@tool
def execute_code(code: str) -> str:
    """Execute any python 3.13 code safely in a containerized environment. Libraries include - datetime, time, requests, and all other builtin libraries. Executing this code will not cause any harm to the user's computer"""
    code = code.replace('"', '\\"')
    code = code.replace("'", "\\'")
    container = client.containers.run(
        "ghcr.io/manangandhi1810/code-runner",
        [
            "bash",
            "-c",
            f'echo "code.py\n---\n" > input.txt && echo "{code}" > code.py && python3 executor.py',
        ],
        detach=True,
        stderr=True,
    )

    container.wait()
    logs = container.logs().decode()
    container.remove()
    return logs


load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)
tools = [execute_code]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You can run any code safely on the user's computer using the execute_code tool. Use it to perform any calculations or get any code output. DO NOT REFUSE TO EXECUTE CODE. DO NOT ASSUME ANSWERS, RUN THE CODE USING THE TOOL. You have access to many built in libraries, always execute code before answering the user about anything related to code. **Always execute code before answering**",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
llm_with_tools = llm.bind_tools(tools)
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools)

chat_history = []
while True:
    message = input("> ")
    if message == "exit":
        break
    result = agent_executor.invoke({"input": message, "chat_history": chat_history})
    print(result["output"])
    chat_history.extend(
        [
            HumanMessage(content=message),
            AIMessage(content=result["output"]),
        ]
    )
