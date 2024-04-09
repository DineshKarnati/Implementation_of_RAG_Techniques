# from langchain_community.chat_models import ChatOpenAI
# from langchain.chains import GraphCypherQAChain
# # from langchain_community.graphs import Neo4jGraphStorage
# from credentials import uri, username, password, OPENAI_API_KEY
# from neo4j import GraphDatabase, basic_auth
from credentials import uri, username, password, OPENAI_API_KEY
# OPENAI_API_KEY = ""
import os
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# driver = GraphDatabase.driver(uri, auth=basic_auth(username, password))
#
# # graph = Neo4jGraphStorage(
# #     url=uri, username=username, password=password
# # )
# # print(driver.schema)
#
#
# chain = GraphCypherQAChain.from_llm(
#     ChatOpenAI(temperature=0), graph=driver, verbose=True
# )
#
# graph_result = chain.run("Who were the siblings of Leonhard Euler?")


# Core libraries
import langchain

# OpenAI library (assuming you're using the official OpenAI Python client)
import openai

# Langchain Neo4j integration
from langchain_community.graphs import Neo4jGraph
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Initialize OpenAI API (replace "YOUR_OPENAI_API_KEY" with your actual key)
openai.api_key = OPENAI_API_KEY

# Define LLM agent (assuming OpenAI)
llm = OpenAI()

# Connect to Neo4j database (replace placeholders with your connection details)
graph = Neo4jGraph(
    url=uri,
    username=username,
    password=password
)
from langchain.vectorstores.neo4j_vector import Neo4jVector

neo4j_vector_store = Neo4jVector.from_documents(
    documents,
    OpenAIEmbeddings(),
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)
retriever = neo4j_vector_store.as_retriever()

chain = RetrievalQAWithSourcesChain.from_chain_type(
    OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)

# Ask a question
question = "Who are the actors in the movie 'Ice and Fire'?"
answer = chain(
    {"question": question},
    return_only_outputs=True,
)
print(answer["answer"])

