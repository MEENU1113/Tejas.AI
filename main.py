import os
from langchain.prompts import PromptTemplate
from langchain_community.llms import Replicate
from neo4j import GraphDatabase
import replicate
import streamlit as st
import pandas as pd

os.environ["REPLICATE_API_TOKEN"] = "r8_YWRQZxYvxEt3vvTjDVrebE5KZrWELsq1x9Rba"

llm = Replicate(
    model="meta/llama-2-70b-chat",
    model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
)

df = pd.read_csv("https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv")

st.title(" NLQBase (Natural Language Query Base) ")
st.write("The Dataset which was choosen for the demonstration is : ")
st.write(df)

# Initialize session state to keep track of the application's state
if 'queries' not in st.session_state:
    st.session_state.queries = []

# Define the prompt template with the schema
prompt_input = """
Given the following schema:
Node properties:
Movie {imdbRating: FLOAT, id: STRING, released: DATE, title: STRING}
Person {name: STRING}
Genre {name: STRING}
Relationship properties:

The relationships:
(:Movie)-[:IN_GENRE]->(:Genre)
(:Person)-[:DIRECTED]->(:Movie)
(:Person)-[:ACTED_IN]->(:Movie)

Provide me only the Cypher query not any other text for the following natural language request. Provide me only the correct and only query.
"""

# Function to generate Cypher query from natural language query using LLM
def get_cypher_query(user_query):
    full_prompt = prompt_input + f"\n{user_query}"
    cypher_query = llm(full_prompt)
    return cypher_query

# Function to run Cypher query on Neo4j
def run_cypher_query(query):
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        result = session.run(query)
        return result.data()

# Display previous queries and results
for query, results, nl_response in st.session_state.queries:
    st.write(nl_response)

# User input for natural language query
user_query = st.text_input("Enter your query")

if user_query:

    cypher_query = get_cypher_query(user_query)

    query_results = run_cypher_query(cypher_query)
   
    prompt_to_nl = f"The query by the user is: {user_query}\nThe exact answer is:\n{query_results}\nGive me the above response in natural language only like an answer to a question."
    natural_language_response = llm(prompt_to_nl)

    st.write("The answer for the given query is:")
    st.write(natural_language_response)

    # Store the query and results in session state for repeated display
    st.session_state.queries.append((user_query, query_results, natural_language_response))