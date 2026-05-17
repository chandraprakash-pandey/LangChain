from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.7)

st.header("Research Assistant")

paper_input = st.selectbox("Select a research paper:", ["Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners"])

style_input = st.selectbox("Select a summarization style:", ["Bullet points", "Paragraph summary", "Key takeaways", "code snippets with explanations"])

length_input = st.slider("Select the length of the summary:", min_value=50, max_value=500, value=150)

template = load_prompt("template.json")

# prompt = template.invoke(
#     {'paper_input':paper_input,
#     'style_input':style_input,
#     'length_input':length_input}
# )

if st.button("Summarize"):
    # st.write("Hello")
    # result = model.invoke(prompt)

    chain = template | model
    result = chain.invoke({'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input})
    
    st.write(result.content)