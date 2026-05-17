from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke on {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model = 'models/gemini-2.5-flash')

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following Joke - {joke}',
    input_variables=['joke']
)

joke_generator_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

result = final_chain.invoke({'topic': 'AI'})

print(result['joke'])

print(result['explanation'])