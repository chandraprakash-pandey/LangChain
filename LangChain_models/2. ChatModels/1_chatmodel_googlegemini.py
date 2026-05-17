from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=1.6)

response = model.invoke("tell me a joke on the topic of AI")
print(response.content)