from langchain_community.document_loaders import  WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'models/gemini-2.5-flash')

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url = 'https://www.amazon.com/Dell-Laptop-DC16251-16-0-inch-Touchscreen-Processor/dp/B0G8K5H16P/ref=sr_1_8?sr=8-8'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'question': 'What is the basic onsite service of Dell 16 Laptop DC16251-16.0-inch 16:10 2K Touchscreen Display, Intel Core 7 150U Processor, 16GB DDR5 RAM, 1TB SSD, Intel Graphics, Windows 11 Home,', 'text': docs[0].page_content})

print(result)