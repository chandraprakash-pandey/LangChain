from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'models/gemini-2.5-flash')

prompt = PromptTemplate(
    template='Wrtie the summary of the following information \n {info}',
    input_variables=['info']
)

parser = StrOutputParser()

loader = TextLoader("cricket.txt", encoding="utf-8")

docs = loader.load()

# print(docs)
# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({'info': docs[0].page_content})

print(result)