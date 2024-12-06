from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import streamlit as st

template = "You are a social media manager who writes social media posts on the topic provided as {val}."
prompt_template = PromptTemplate(
    input_variables=["val"], template=template
)
llm = ChatOllama(model="llama3.1", temperature=0)
chain = prompt_template | llm | StrOutputParser()

st.title("Social Media Post Generator")
st.write("Enter a topic to generate a social media post.")

user_input = st.text_input("Topic:", placeholder="Enter a topic, e.g., Weather")

if user_input:
    with st.spinner("Generating social media post..."):
        out = chain.invoke(input={"val": user_input})
        st.write("### Generated Post:")
        st.write(out)