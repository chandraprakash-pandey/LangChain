from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
)

model = ChatHuggingFace(llm = llm)

#1st Prompt -> Detailed Explaination
template1 = PromptTemplate(
    template='Write a deailed report on {topic}',
    input_variables=['topic']
)


#2nd Prompt-> SUmmary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})

result1 = model.invoke(prompt2)

print(result1.content)