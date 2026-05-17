from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=1.6)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me a joke."),
]

response = model.invoke(messages)
messages.append(AIMessage(content=response.content))

print(messages)