from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity 

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

docs =[
    "Sachin Tendulkar is a former Indian cricketer",
    "Virat Kohli is the current captain of the Indian cricket team",
    "M.S. Dhoni is a former Indian cricketer and captain",
    "Malinga is a former Sri Lankan cricketer",
    "Kris gayle is a former West Indian cricketer"
]

# query = "Who is the current captain of the Indian cricket team?"

query = input("Enter Your Query: ")


doc_embeddings = [embedding.embed_query(doc) for doc in docs]
query_embedding = embedding.embed_query(query)

print("Number of doc embeddings:", len(doc_embeddings))
print("Shape of each:", len(doc_embeddings[0]))

print(str(doc_embeddings))

scores = cosine_similarity([query_embedding], doc_embeddings)
print("Similarity scores: ", scores)


index, score = sorted(list(enumerate(scores[0])), key=lambda x: x[1])[-1]

print(docs[index])
print("Similarity score is: ",score)