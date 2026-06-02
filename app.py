import os
import sqlite3
import gradio as gr

# LangChain Document Handlers
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Infrastructure & Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Core Engine Prompts & Chains
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# LangGraph Agentic Layer & Memory
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

# ==========================================
# 1. RETRIEVAL PIPELINE (PDF LOADING & FAISS)
# ==========================================
print("Processing rehabilitation manuals...")

# Define paths to your target patent documents
loaders = [
    PyPDFLoader("3D Robotic Elbow Rehabilitation Stimulator.pdf"),
    PyPDFLoader("3D Robotic Shoulder Rehabilitation Stimulator V2.pdf"),
    PyPDFLoader("3D Robotic Wrist Rehabilitation Stimulator.pdf")
]

docs = []
for loader in loaders:
    try:
        docs.extend(loader.load())
    except Exception as e:
        print(f"Warning: Could not load a file. Details: {e}")

# High-fidelity chunking strategy optimized for engineering steps
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", "Step", "Calibration", "."]
)
chunks = text_splitter.split_documents(docs)
print(f"Extracted {len(chunks)} contextual chunks successfully.")

# Initialize embedding model and index data into FAISS
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = FAISS.from_documents(chunks, embedding_model)
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
print("Vector storage engine generated successfully.")

# ==========================================
# 2. LLM INITIALIZATION & PROMPT ARCHITECTURE
# ==========================================
if "GROQ_API_KEY" not in os.environ:
    print("Warning: 'GROQ_API_KEY' environment variable not detected. Ensure it is set in your environment.")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_retries=2
)

# Advanced Institutional Knowledge Prompt
template = """
You are an advanced rehabilitation engineering assistant specialized in:
- robotic rehabilitation systems
- upper-limb rehabilitation
- shoulder rehabilitation
- elbow rehabilitation
- wrist rehabilitation
- telekinematic rehabilitation workflows
- rehabilitation biomechanics
- therapy guidance
- rehabilitation safety procedures
- rehabilitation monitoring
- web-based rehabilitation systems
- rehabilitation engineering architectures
- interactive rehabilitation feedback systems

Use ONLY the provided rehabilitation manual context.

CRITICAL PRIVACY RULES:
- Do not provide personal information.
- Do not provide inventor information.
- Do not provide author information.
- Do not provide developer information.
- Do not provide ownership information.
- Do not provide biographical information.
- Do not reveal names of individuals associated with the system.
- Never output names found in document metadata, headers, footers, acknowledgements, references, patents, publications, or authorship sections.
- If asked who invented, developed, authored, created, owns, or designed the system,
  respond only with the technical purpose, rehabilitation workflow,
  engineering architecture, and clinical utility described in the context.

Instructions:
- Answer technically and professionally.
- Focus on rehabilitation workflows, rehabilitation engineering, therapy guidance, and clinical utility.
- Explain rehabilitation procedures clearly and systematically.
- If remote rehabilitation or tele-rehabilitation is mentioned, explain it as a telekinematic rehabilitation workflow.
- When architecture, workflow, monitoring, or system-design questions are asked, provide structured diagrammatic-style explanations using: system blocks, workflow pipelines, monitoring layers, rehabilitation stages, feedback loops, control-flow structures, and clinical interaction flow.
- Generate clean architecture-style textual representations when appropriate.
- Explain rehabilitation monitoring, progress tracking, safety workflows, therapy guidance, and rehabilitation control mechanisms clearly.
- Avoid generic healthcare explanations.
- Do not invent hardware or sensors unless explicitly mentioned in the context.
- If the answer is partially available, answer only from the available rehabilitation context.

Context:
{context}

Question:
{question}

Technical Answer:
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | ChatPromptTemplate.from_template(template)
    | llm
    | StrOutputParser()
)

# ==========================================
# 3. CLINICAL DATABASE INFRASTRUCTURE
# ==========================================
DB_FILE = "upper_limb_rehabilitation_suite.db"

def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS resources")
    cur.execute("DROP TABLE IF EXISTS bookings")
    
    cur.execute("""
    CREATE TABLE resources (
        resource_id INTEGER PRIMARY KEY,
        resource_name TEXT,
        status TEXT
    )
    """)
    
    cur.execute("""
    CREATE TABLE bookings (
        booking_id INTEGER PRIMARY KEY,
        patient_name TEXT,
        resource_id INTEGER,
        FOREIGN KEY(resource_id) REFERENCES resources(resource_id)
    )
    """)
    
    cur.executemany("""
    INSERT INTO resources (resource_id, resource_name, status)
    VALUES (?, ?, ?)
    """, [
        (1, "Robotic Shoulder Rehabilitation Stimulator", "available"),
        (2, "Robotic Elbow Rehabilitation Stimulator", "available"),
        (3, "Robotic Wrist Rehabilitation Stimulator", "available"),
        (4, "Upper Limb Physiotherapy Unit", "available"),
        (5, "Motion Analysis and Rehabilitation Lab", "available")
    ])
    conn.commit()
    conn.close()

init_db()
print("Relational schema and clinical assets database operational.")

# ==========================================
# 4. AGENTIC TOOLS SETUP
# ==========================================
@tool
def rehab_knowledge_tool(question: str) -> str:
    """
    Use ONLY for technical questions about upper-limb rehabilitation stimulators,
    calibration, sensors, therapy workflow, safety precautions,
    rehabilitation exercises, and simulator manuals.
    """
    return rag_chain.invoke(question)

@tool
def check_availability_tool() -> str:
    """Use ONLY for checking available upper-limb rehabilitation resources."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT resource_id, resource_name FROM resources WHERE status='available'")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return "No upper-limb rehabilitation resources available."
    return "\n".join([f"Resource {r[0]}: {r[1]}" for r in rows])

@tool
def book_resource_tool(patient_name: str, resource_id: int) -> str:
    """Use ONLY when the user explicitly requests to book an upper-limb rehabilitation resource."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM resources WHERE resource_id=? AND status='available'", (resource_id,))
    resource = cursor.fetchone()
    if not resource:
        conn.close()
        return f"Resource {resource_id} is currently occupied or non-existent."
    
    cursor.execute("INSERT INTO bookings (patient_name, resource_id) VALUES (?, ?)", (patient_name, resource_id))
    cursor.execute("UPDATE resources SET status='occupied' WHERE resource_id=?", (resource_id,))
    conn.commit()
    conn.close()
    return f"Upper-limb rehabilitation resource {resource_id} successfully booked for {patient_name}."

tools = [rehab_knowledge_tool, check_availability_tool, book_resource_tool]

system_message = """
You are an AI-Powered Unified Upper Limb Rehabilitation Suite Assistant.
Available tools:
1. rehab_knowledge_tool -> technical rehabilitation manual insights
2. check_availability_tool -> tracking clinical asset availability
3. book_resource_tool -> processes bookings when patient_name and resource_id are provided.
"""

class State(TypedDict):
    messages: Annotated[list, add_messages]

agent = create_react_agent(llm, tools, prompt=system_message)
graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph_builder.set_entry_point("agent")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# ==========================================
# 5. INTENT ROUTER LOGIC
# ==========================================
def healthcare_suite_router(user_query):
    query = user_query.lower()
    if "book" in query:
        return "Resource booking functionality is currently under development."
    elif "available" in query or "availability" in query:
        return check_availability_tool.invoke({})
    elif any(x in query for x in ["motion analysis", "physiotherapy", "rehabilitation lab", "resource"]):
        return """
**Motion Analysis and Rehabilitation Lab**
A specialized upper-limb rehabilitation facility designed to evaluate
arm movement, joint coordination, rehabilitation performance,
range of motion, and biomechanical therapy progression.
"""
    else:
        return rehab_knowledge_tool.invoke({"question": user_query})

# ==========================================
# 6. GRADIO GRAPHICAL UI RUNTIME
# ==========================================
def respond(message, history):
    try:
        if isinstance(message, dict):
            user_message = message.get("content", "")
        else:
            user_message = str(message)
            
        result = healthcare_suite_router(user_message)
        return {"role": "assistant", "content": str(result)}
    except Exception as e:
        return {"role": "assistant", "content": f"Core Execution Exception: {str(e)}"}

description_html = """
<div style="text-align:center;">
    <h3>Built by Dr. Lakshmi Gandi</h3>
    <p>Advanced Clinical Intelligence Platform: Multi-Turn RAG Orchestration Engine</p>
</div>
"""

demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="🏥 Unified Upper Limb Rehabilitation Suite",
    description=description_html,
    examples=[
        "How does the robotic shoulder rehabilitation stimulator work?",
        "Compare the elbow and wrist rehabilitation stimulators.",
        "What safety precautions are mentioned in the manuals?",
        "Which upper-limb rehabilitation resources are available?",
        "What is the Motion Analysis and Rehabilitation Lab?"
    ]
)

if __name__ == "__main__":
    # Deploy app securely with sharing enabled
    demo.launch(share=True)
