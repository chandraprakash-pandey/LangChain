from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI


# Chat Template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support Agent'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

# Load chat history
chat_history = []

with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

print(chat_history)

#create prompt
prompt = chat_template.invoke({'chat_history': chat_history , 'query': "Where is my refund"})

model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

respond = model.invoke(prompt)

print(respond)

# print(prompt)