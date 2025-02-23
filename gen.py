import os
import fitz
import requests
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import Dict, List, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

app = FastAPI()

# Global FAISS index and text storage
faiss_index = None
texts_db = None

# ----------------------------
# PDF Handling Functions
# ----------------------------
def download_pdf(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        file_path = "downloaded.pdf"
        with open(file_path, "wb") as file:
            file.write(response.content)
        return file_path
    raise HTTPException(status_code=400, detail="Failed to download PDF")

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)

def create_faiss_index(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    global texts_db
    texts_db = texts  # Save texts for later retrieval if needed
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    global faiss_index
    faiss_index = FAISS.from_texts(texts, embeddings)

# ----------------------------
# FastAPI Endpoints
# ----------------------------
class IngestRequest(BaseModel):
    pdf_url: str

@app.post("/ingest/")
def ingest_pdf(request: IngestRequest):
    pdf_path = download_pdf(request.pdf_url)
    text = extract_text_from_pdf(pdf_path)
    create_faiss_index(text)
    return {"message": "PDF successfully ingested"}

class AskRequest(BaseModel):
    query: str

@app.post("/ask/")
def ask_question(request: AskRequest):
    if faiss_index is None:
        raise HTTPException(status_code=400, detail="No FAISS index found. Ingest a PDF first.")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embeddings.embed_query(request.query)
    docs = faiss_index.similarity_search_by_vector(query_embedding, k=5)
    retrieved_text = "\n".join(doc.page_content for doc in docs)
    
    # Generate answer using Hugging Face model
    llm = HuggingFacePipeline.from_model_id(
        model_id="facebook/opt-1.3b",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 50}  # Ensure sufficient generation length
    )
    response = llm.invoke(f"Based on this document: {retrieved_text}\n\nAnswer: {request.query}")
    return {"response": response}

# ----------------------------
# Tool-Based Extensions (LangGraph)
# ----------------------------
OPENWEATHERMAP_API_KEY = "8bdc28c2e44accfda2a550b7aff0fc14"
NEWSAPI_API_KEY = "764118ef0c7a4eb79aa855aa292da6be"

@tool
def weather(location: str) -> str:
    """Fetches current weather for a given location."""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
    response = requests.get(url).json()
    if "main" in response:
        temp = response['main']['temp']
        desc = response['weather'][0]['description']
        return f"Current weather in {location}: {temp}°C, {desc.capitalize()}."
    return "Weather data unavailable."

@tool
def news(topic: str) -> str:
    """Fetches top 5 news articles related to a given topic."""
    url = f"https://newsapi.org/v2/top-headlines?q={topic}&apiKey={NEWSAPI_API_KEY}"
    response = requests.get(url).json()
    articles = response.get("articles", [])[:5]
    return "\n".join([f"{a['title']} - {a['source']['name']}" for a in articles]) if articles else "No news found."

# ----------------------------
# Agent Graph Functions
# ----------------------------
class GraphState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    location: Optional[str]
    topic: Optional[str]

def should_use_tool(state: GraphState) -> Dict:
    messages = state["messages"]
    last_message = messages[-1].content.lower()
    if "weather in" in last_message:
        location = last_message.split("weather in ")[-1].split("?")[0].strip()
        return {"next_step": "use_weather_tool", "location": location}
    elif "news about" in last_message:
        topic = last_message.split("news about ")[-1].split("?")[0].strip()
        return {"next_step": "use_news_tool", "topic": topic}
    else:
        return {"next_step": "generate_response"}

def use_weather_tool(state: GraphState) -> Dict:
    result = weather.invoke(state["location"])
    
    # Ensure only one weather message is present
    return {"messages": [AIMessage(content=result)], "next_step": "end"}


def use_news_tool(state: GraphState) -> Dict:
    result = news.invoke(state["topic"])

    # Only keep the latest news response
    return {"messages": [AIMessage(content=result)], "next_step": "generate_response"}


def generate_response(state: GraphState) -> Dict:
    # Instantiate the pipeline with the desired generation config
    llm = HuggingFacePipeline.from_model_id(
        model_id="facebook/opt-1.3b",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 50}  # Increase generation length
    )
    
    response = llm.invoke(f"{state['messages'][-1].content}")

    # Only keep the latest response
    return {"messages": [AIMessage(content=response)], "next_step": "end"}


def create_agent_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("should_use_tool", should_use_tool)
    workflow.add_node("use_weather_tool", use_weather_tool)
    workflow.add_node("use_news_tool", use_news_tool)
    workflow.add_node("generate_response", generate_response)
    workflow.add_conditional_edges(
        "should_use_tool",
        lambda x: x["next_step"],
        {
            "use_weather_tool": "use_weather_tool",
            "use_news_tool": "use_news_tool",
            "generate_response": "generate_response"
        }
    )
    workflow.add_edge("use_weather_tool", "generate_response")
    workflow.add_edge("use_news_tool", "generate_response")
    workflow.set_entry_point("should_use_tool")
    return workflow.compile()

graph = create_agent_graph()

class AgentRequest(BaseModel):
    messages: List[dict]

@app.post("/agent/")
async def agent_query(request: AgentRequest):
    formatted_messages = [
        HumanMessage(content=msg["content"]) if msg["type"] == "human" else AIMessage(content=msg["content"])
        for msg in request.messages
    ]
    initial_state = {"messages": formatted_messages, "next_step": None}
    final_state = graph.invoke(initial_state)
    final_message = final_state["messages"][-1].content if final_state["messages"] else "No response generated"
    return {"response": final_message}

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy"}
