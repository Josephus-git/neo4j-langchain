import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain

# Initialize the LLM
model = init_chat_model("gpt-5.2", model_provider="openai")

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
)

# Seperate cypher model
cypher_model = init_chat_model(
    "gpt-5-mini",
    model_provider="openai",
    reasoning={"effort": "high"},
)

# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=model,
    cypher_llm=cypher_model,
    allow_dangerous_requests=True,
    verbose=True,
)

# Invoke the chain
question = "How many movies are in the Sci-Fi genre?"
response = cypher_qa.invoke({"query": question})
print(response["result"])

