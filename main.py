# Import Libraries
import base64
import os
import streamlit as st
import uuid
from datetime import datetime
from openai import OpenAI
from langfuse.openai import openai
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import json

# Load API keys from secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE_SECRET_KEY"]
LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_BASE_URL = st.secrets["LANGFUSE_BASE_URL"]

lf_handler = CallbackHandler()

# Langfuse untuk event manual (SDK murni)
lf_client = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_BASE_URL
    )
# completion = openai.chat.completions.create(
#   name="test-chat",
#   model="gpt-4o-mini",
#   messages=[
#       {"role": "system", "content": "You are a very accurate calculator. You output only the result of the calculation."},
#       {"role": "user", "content": "1 + 1 = "}],
#   metadata={"someMetadataKey": "someValue"},
# )

# Initialize OpenAI client (for summarization)
client = OpenAI(
    api_key=OPENAI_API_KEY
    )

# creating a connection (object) to an OpenAI language model (LLM)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key = OPENAI_API_KEY,
    temperature=0.5,
    callbacks=[lf_handler]
    )

#Embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key = OPENAI_API_KEY
    )

#RAG (call document from Qdrant)
collection_name = "IMDB_documents2"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
    )

# Get current date and time
now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")

# Define Tools
@tool
def get_relevant_docs(question):
    """This is your main tool. Use this tool to get relevant documents about movies from database."""
    results = qdrant.similarity_search(
      question,
      k=10
      )
    return [{
        "title": doc.metadata.get("title", "Unknown Title"),
        "IMDB_rating": doc.metadata.get("IMDB_rating", "N/A"),
        "director": doc.metadata.get("Director", "Unknown"),
        "genre": doc.metadata.get("genre", "Unknown"),
        "summary": doc.page_content[:500]}
    for doc in results]

# Tool for searching the web
@tool
def web_search(query, num_results=3):
    """This is your secondary tool. Use it if the database cannot provide information.
    Searches information about movies only from Wikipedia, Rottentomatoes, and IMDb website for reliable movie-related information."""
    results = []
    restricted_query = f"{query} site:wikipedia.org OR site:imdb.com OR site:rottentomatoes.com"
    
    with DDGS() as ddgs:
        for r in ddgs.text(restricted_query, max_results=num_results):
            # Filter results to ensure only IMDb or Wikipedia links are returned
            link = r.get("href", "")
            if "wikipedia.org" in link or "imdb.com" in link or "rottentomatoes.com" in link:
                results.append({
                    "title": r.get("title"),
                    "link": link,
                    "snippet": r.get("body")
                })
    return results

@tool
def fetch_webpage_text(url, max_chars=1500):
    """Use this to Fetch and clean text from a web page that found by web_search tool."""
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        return text[:max_chars]
    except Exception:
        return ""

@tool
def summarize_text(context, question):
    """Use this to Summarize web information into a clear and concise answer."""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes result of fetch_webpage_text tool."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nSummarize the key points clearly and concisely."}
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content

tools = [get_relevant_docs, web_search, fetch_webpage_text, summarize_text]

def chat_robot(question, history):
    agent = create_react_agent(
        model=llm,
        tools=tools
        )
    
    messages = [{"role": "system", "content": """
            You are a friendly and helpful AI movie assistant 🎥🍿.
            The current date and time is: {current_time}.
            Always respond in the same language as the user.
            You are knowledgeable about films, actors, directors, and box office data.
            Use emojis and emoticons naturally to make your responses fun and expressive 😄✨.
            Find information about movies first in database using get_relevant_docs tool.
            If database can't provide information needed to response user, use web_search tool to retrieve information from internet (wikipedia.org, imdb.com, rottentomatoes.com only).
            Use fetch_webpage_text and summarize_text tools to summarize website information from web_search tool.
            Movies recommendation or list have to sort by IMDB_rating (descending).
            Use context and chat history to provide coherent answers.
            Use markdown (like bullet points or lists) for clarity.
            Avoid saying things like 'according to the provided context'.
            Be concise but informative.
            Only answer movie-related questions.
            Add related links at the bottom when relevant 🌐."""}]

    if history.strip():
        messages.append({"role": "assistant", "content": f"Conversation history:\n{history}"})

    messages.append({"role": "user", "content": question})

    result = agent.invoke({"messages": messages})
    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0

    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
        elif "token_usage" in message.response_metadata:
            # Fallback for older or different structures
            total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

    price = 17_000*(total_input_tokens*0.15 + total_output_tokens*0.6)/1_000_000

    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_message_content = message.content
            tool_messages.append(tool_message_content)
 
    response = {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }
    return response
    
def set_gif_background(gif_path):
    # Ensure the file exists
    if not os.path.exists(gif_path):
        st.error(f"❌ File not found: {gif_path}")
        return

    # Read and encode the GIF
    with open(gif_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    # Inject CSS with base64 image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/gif;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

gif_path = r"C:\Users\timbu\Downloads\Purwadhika\04. Capstone Project\Caps3\Capstone3_Final\nodes.gif"
set_gif_background(gif_path)

st.title("Chatbot 🎥 Movies Dictionary")
st.write("🤖 Welcome! Let’s find any information about movies together.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about any movie 🎥"):
    messages_history = st.session_state.get("messages", [])[-20:]
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("AI"):
        response = chat_robot(prompt, history)
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "AI", "content": answer})

    with st.expander("**🧩 Tool Calls:**"): #creates a collapsible section with title 🧩 Tool Calls.
        st.code(response["tool_messages"])

    with st.expander("**📜 Chat History:**"):
        st.code(history)

    with st.expander("💰 Token Usage"):
        st.code(
            f'Input tokens: {response["total_input_tokens"]}\n'
            f'Output tokens: {response["total_output_tokens"]}\n'
            f'Estimated cost (Rp): {response["price"]:.2f}'
        )

    # --- 🧠 User Feedback Section ---
    with st.container():
        st.markdown("### 💬 Was this answer helpful?")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("👍 Yes", key=f"yes_{len(st.session_state.messages)}"):
                st.success("Thanks for your feedback! 🙌")
                st.session_state.last_feedback = {"question": prompt, "feedback": "positive", "answer": answer}

        with col2:
            if st.button("👎 No", key=f"no_{len(st.session_state.messages)}"):
                st.warning("Got it! We’ll try to improve.")
                st.session_state.last_feedback = {"question": prompt, "feedback": "negative", "answer": answer}

        with col3:
            custom_feedback = st.text_input("Additional comments (optional):", key=f"comment_{len(st.session_state.messages)}")
            if custom_feedback:
                st.session_state.last_feedback["comment"] = custom_feedback

 