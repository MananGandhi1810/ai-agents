import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from selenium import webdriver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
from markdownify import markdownify

options = webdriver.ChromeOptions()
options.add_argument("--headless")
browser = webdriver.Chrome(options=options)


@tool
def get_url_data(url: str) -> str:
    """Returns Markdown (from HTML source) of the requested URL by rendering Javascript in the browser"""
    url = url.strip().strip("'").strip('"').strip("`")
    if "https://" not in url and "http://" not in url:
        url = "https://" + url
    print("Requesting URL:", url)
    try:
        browser.get(url)
        time.sleep(1)
        data = browser.page_source
        markdown = markdownify(data).strip()
        return markdown
    except Exception as e:
        print(e)
        return e


def format_log_to_messages(intermediate_steps):
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = []
    for action, observation in intermediate_steps:
        thoughts.append(AIMessage(content=action.log))
        human_message = HumanMessage(content=f"Observation: {observation}")
        thoughts.append(human_message)
    return thoughts


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
            """You are a web-browsing agent with access to a Chromium instance
YOU HAVE INTERNET ACCESS SO YOU CAN USE ANY TOOL THAT REQUIRES INTERNET ACCESS
SEARCH GOOGLE IF YOU NEED ANY HELP
If you have multiple steps, query one URL at a time and continue until you get the answer

Answer the following questions as best you can. You have access to the following tools:
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

Begin!""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "User: {input}"),
        ("ai", "Scratchpad: {agent_scratchpad}"),
    ]
)
llm_with_tools = llm.bind_tools(tools)
agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=prompt,
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)

chat_history = []
agent_scratchpad = format_log_to_messages([])
while True:
    message = input("> ")
    if message == "exit":
        break
    result = agent_executor.invoke(
        {
            "input": message,
            "chat_history": chat_history,
        }
    )
    print(result["output"])
    intermediate_steps = str(result["intermediate_steps"])
    output = result["output"]
    chat_history.extend(
        [
            HumanMessage(content=message),
            AIMessage(
                content=f"Intermediate Steps:\n\n{intermediate_steps}\n\nOutput:\n\n{output}"
            ),
        ]
    )
