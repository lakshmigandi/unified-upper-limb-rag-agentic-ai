# 🏥 Unified Upper Limb Rehabilitation Suite: Agentic AI & RAG Engine

An advanced, production-ready clinical intelligence platform built to orchestrate intelligent multi-turn Retrieval-Augmented Generation (RAG) and Agentic reasoning. This system integrates technical data from specialized rehabilitation stimulators with real-time relational clinical assets management.

**Built by:** Dr. Lakshmi Gandi  
**Frameworks Used:** LangGraph, LangChain, FAISS Vector DB, SQLite3, Gradio, ChatGroq (Llama 3.1)

---

## 🚀 Key Features

- **Multi-Document Technical RAG:** Parses, chunks, and indexes specialized clinical patent manuals for Upper-Limb (Shoulder, Elbow, and Wrist) Rehabilitation Stimulators.
- **Agentic Tool Layer:** Built on a LangGraph state network using `create_react_agent` to seamlessly route queries between deep text-retrieval models and local database operations.
- **Relational Asset Tracking:** Utilizes an integrated SQLite architecture to dynamically track available medical resources and manage mock patient bookings securely.
- **Production UI:** Deploys a conversational workspace utilizing Gradio's stateful multi-turn interface layout.

---

## 🏗️ Repository Architecture

The workspace is organized to support quick cloud deployment and comprehensive research review:

```text
unified-upperlimb-rag-agentic-ai/
│
├── 3D Robotic Elbow Rehabilitation Stimulator.pdf       # Technical Source Document
├── 3D Robotic Shoulder Rehabilitation Stimulator V2.pdf  # Technical Source Document
├── 3D Robotic Wrist Rehabilitation Stimulator.pdf       # Technical Source Document
├── app.py                                               # Production Deployment Script
├── requirements.txt                                     # Dependency Specifications
└── research_notebook.ipynb                              # Core Development Playground
