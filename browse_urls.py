import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv


@tool
def get_url_data(url: str) -> str:
    """Returns data by scraping the url"""
    data = requests.get(url).content
    return data


load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
)
tools = [get_url_data]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You can get data about a website or URL, or anything on the internet using get_url_data tool",
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
