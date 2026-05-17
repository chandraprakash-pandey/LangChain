from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=1.6)

chatHistory = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    user_input = input("User :")
    chatHistory.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break

    response = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content=response.content))
    print("AI :", response.content)

print(chatHistory)