from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(
    model="models/gemini-2.5-flash",   # ✅ exact name from your list
)

result = llm.invoke("What is the capital of India?")
print(result)