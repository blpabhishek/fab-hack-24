import os 
import getpass

if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("postgresql://data-insights:data-insights@192.168.176.63:5432/data-insights")
print(db.dialect)
print(db.get_usable_table_names())
# print(db.run("SELECT * FROM \"Customers\" LIMIT 10;"))


from typing_extensions import TypedDict

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4o-mini")


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_experimental.llms.ollama_functions import OllamaFunctions

llm = OllamaLLM(model="duckdb-nsql", temperature=0)
# llm = OllamaFunctions(model="phi3", format="json", temperature=0)


from langchain import hub

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

assert len(query_prompt_template.messages) == 1
query_prompt_template.messages[0].pretty_print()



from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 1000,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    # print(prompt)
    # print(state)
    # structured_llm = llm.with_structured_output(QueryOutput)
    result = llm.invoke(prompt)
    print(result)
    # return {"query": result["query"]}
    return result


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    # return {"result": execute_query_tool.invoke(state["query"])}
    return execute_query_tool.invoke(state)


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()


def main(question):
    ans = write_query({"question": question})
    print("-----------------")
    print(ans)

    result = execute_query(ans)
    return result

# ans = write_query({"question": "How many Employees are there?"})
# print(ans)
# ans = execute_query({"query": "SELECT COUNT(EmployeeId) AS EmployeeCount FROM Employee;"})
# print(ans)


# for step in graph.stream(
#     {"question": "How many customers are there?"}, stream_mode="updates"
# ):
#     print(step)



from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import streamlit as st

# template = "You are a social media manager who writes social media posts on the topic provided as {val}."
# prompt_template = PromptTemplate(
#     input_variables=["val"], template=template
# )
# llm = ChatOllama(model="llama3.1", temperature=0)
# chain = prompt_template | llm | StrOutputParser()

# st.title("Customer Database Lookup")
st.write("Query your database")

user_input = st.text_input("Topic:", placeholder="Enter a topic, e.g., Weather")

if user_input:
    with st.spinner("Generating social media post..."):
        # out = chain.invoke(input={"val": user_input})
        out = main(user_input)
        # chain.invoke(input={"val": user_input})
        st.write("### Generated Post:")
        st.write(out)