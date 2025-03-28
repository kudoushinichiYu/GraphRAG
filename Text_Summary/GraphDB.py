import os
import re
import torch
from typing import List
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.output_parsers import JsonOutputParser

# ========== 1. Setup Neo4j Connection ==========
with open('D:/columbia25spring/huggingfacecode.txt','r') as text:
    api_token = text.read()
    login(api_token)
with open('D:/huggingface_cache/Neo4j-91aec2fb-Created-2025-03-26.txt','r') as text2:
    content = text2.read()
    uri_match = re.search(r'NEO4J_URI=(\S+)', content).group(1)
    username_match = re.search(r'NEO4J_USERNAME=(\S+)', content).group(1)
    password_match = re.search(r'NEO4J_PASSWORD=(\S+)', content).group(1)
    print(uri_match)

os.environ["NEO4J_URI"] = uri_match
os.environ["NEO4J_USERNAME"] = username_match
os.environ["NEO4J_PASSWORD"] = password_match

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)

# ========== 2. Load LLM (Meta Llama 3) ==========
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
llm = HuggingFacePipeline(pipeline=pipe)

# ========== 3. Load Wikipedia Data ==========
raw_documents = WikipediaLoader(query="Russo-Ukrainian War").load()
print(f"Loaded {len(raw_documents)} raw documents")

# ========== 4. Split Text into Chunks ==========
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
documents = text_splitter.split_documents(raw_documents)
print(f"Split into {len(documents)} chunks")

# ========== 5. Extract Graph Data ==========
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents[:10])
print(f"Extracted {len(graph_documents)} graph documents")
graph.add_graph_documents(graph_documents, include_source=True)

# ========== 6. Setup Vector Index in Neo4j ==========
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)
vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# ========== 7. Entity Extraction Chain ==========
class Entities(BaseModel):
    names: List[str] = Field(..., description="Extracted entities from text.")
    relations: List[str] = Field(default_factory=list, description="Extracted relations from text.")

# Define the output parser
parser = JsonOutputParser(pydantic_object=Entities)

# Create the prompt with format instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert entity extraction system. Extract all organization and people entities from the given text.
     {format_instructions}"""),
    ("human", "Text: {question}"),
]).partial(format_instructions=parser.get_format_instructions())

# Create the chain
entity_chain = prompt | llm | parser

print("Creating fulltext index...")
graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]")

# ========== 8. Full-Text Search in Neo4j ==========
def generate_full_text_query(input: str) -> str:
    return " OR ".join([f"{word}~" for word in input.split()])

# ========== 9. Structured Retrieval from Graph ==========
def structured_retriever(question: str) -> str:
    result = ""
    try:
        # Get entities from LLM
        entities = entity_chain.invoke({"question": question})
        print(f"Extracted entities: {entities}")
        
        if not entities.get('names'):
            return "No entities found"
            
        for entity in entities['names']:
            response = graph.query(
                """
                CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node, score
                MATCH (node)-[r:MENTIONS]->(neighbor)
                RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                UNION
                MATCH (node)<-[r:MENTIONS]-(neighbor)
                RETURN neighbor.name + ' - ' + type(r) + ' -> ' + node.name AS output
                LIMIT 50
                """,
                {"query": generate_full_text_query(entity)}
            )
            result += "\n".join([el['output'] for el in response]) + "\n"
    except Exception as e:
        print(f"Error in structured_retriever: {e}")
        return f"Error retrieving structured data: {e}"
    
    return result.strip()

# ========== 10. Hybrid Retriever (Graph + Vector) ==========
def retriever(question: str) -> str:
    structured_data = structured_retriever(question)
    unstructured_results = vector_index.similarity_search(question, k=3)
    unstructured_data = "\n".join([el.page_content for el in unstructured_results])
    
    final_data = f"""=== Structured Data ===\n{structured_data}\n\n
=== Unstructured Data ===\n{unstructured_data}"""
    return final_data

# ========== 11. Test the System ==========
if __name__ == "__main__":
    question = "What are the key entities involved in the Russia-Ukraine conflict?"
    print("\nProcessing question:", question)
    answer = retriever(question)
    print("\n===== FINAL RESULT =====\n")
    print(answer)