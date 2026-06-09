# WE Smart Assistant 🤖

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)[![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)](https://python.langchain.com/)[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://www.trychroma.com/)[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-black.svg)](https://openai.com/)

**WE Smart Assistant** is an advanced, agentic Retrieval-Augmented Generation (RAG) chatbot designed specifically for **WE Telecom Egypt**. It acts as an intelligent, bilingual (Arabic/English) digital representative, capable of handling complex customer inquiries, providing accurate service details, and navigating WE's extensive knowledge base with high precision.

---

## 🌟 Key Features

### 1. Interactive Streamlit User Interface

WE Smart Assistant comes with a rich, interactive web interface built using Streamlit, offering a seamless chat experience:

- **Bilingual Interface**: The UI dynamically adapts to Arabic and English, mirroring the chatbot's language capabilities.

- **Custom Theming**: Features a custom, branded theme with WE Telecom Egypt's colors and fonts (Cairo, Outfit) for a polished look.

- **File and Image Upload**: Users can upload `.txt`, `.pdf`, `.docx` files, and images (`.png`, `.jpg`, `.jpeg`) directly into the chat. The system automatically extracts text from these documents and images (using EasyOCR) to augment the RAG process.

- **Clear Chat Functionality**: A dedicated button in the sidebar and at the bottom of the chat allows users to clear the conversation history.

- **Welcome Screen**: An engaging welcome screen with suggested quick-start chips guides new users.

- **Real-time Responses**: Displays responses in a streaming fashion, enhancing user engagement.

### 2. Advanced Hybrid RAG Pipeline

The core of the assistant is a sophisticated retrieval system that ensures high accuracy and relevance:

- **Dense Retrieval**: Utilizes `BAAI/bge-m3` embeddings stored in **ChromaDB** for deep semantic understanding.

- **Sparse Retrieval**: Implements **BM25** for exact keyword matching, crucial for specific telecom terms, USSD codes, and service names.

- **Ensemble & Reranking**: Combines both retrieval methods and applies a Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`) via `FlagEmbedding` to surface the most relevant context.

### 3. Intelligent Query Routing & Intent Stripping

The system doesn't just search; it understands the user's intent:

- **Metadata-Aware Routing**: Uses regex-based signals to detect specific domains (e.g., 5G, WE Air, USSD Codes, Payments, "Ma3ak" service for the deaf) and dynamically filters the vector database to narrow down the search space.

- **Intent Stripping**: Automatically removes conversational filler words (e.g., "I want to ask about...") to extract the core search query, improving retrieval accuracy.

- **Ambiguity Resolution**: Analyzes chat history to condense ambiguous follow-up questions (e.g., "How much is it?") into standalone, context-rich queries.

### 4. Bilingual & Culturally Aware Persona

The "WE-Bot" persona is designed to reflect the Egyptian identity:

- **Dynamic Language Switching**: Automatically detects if the user is speaking Arabic (Standard or Egyptian dialect) or English, and responds in the exact same language.

- **Egyptian Dialect Support**: Capable of understanding and responding in polite Egyptian Arabic (e.g., "يا فندم", "تحت أمرك").

- **Strict Hallucination Control**: Programmed to strictly adhere to the provided context. If information is missing, it gracefully directs the user to official WE channels (111 or te.eg) rather than guessing.

### 5. Automated Database Synchronization

The system includes a smart sync mechanism that compares the raw JSON data with the ChromaDB vector store. It automatically detects new, updated, or deleted documents based on content hashing, ensuring the knowledge base is always up-to-date without requiring a full rebuild.

---

## 🏗️ Architecture & Tech Stack

### Tech Stack

- **Orchestration**: LangChain (Core, Community, Classic)

- **LLM**: OpenAI `gpt-4o-mini` (via OpenRouter API)

- **Embeddings**: HuggingFace `BAAI/bge-m3`

- **Reranker**: FlagEmbedding `BAAI/bge-reranker-v2-m3`

- **Vector Database**: ChromaDB

- **Data Processing**: Pydantic, hashlib, regex

### Data Schema

The knowledge base is powered by a highly structured JSON file (`processed_all_no_branches.json`). Each document contains:

- `id`: Unique identifier.

- `text`: The actual content (Service details, pricing, FAQs, etc.).

- `metadata`: Rich filtering tags including `domain`, `sub_domain`, `type`, `category`, `payment_type`, and `display_hint`.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher

- An OpenRouter API Key (for accessing the LLM)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd we-smart-assistant
```

### 2. Install Dependencies

Install the required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory and configure the following variables:

```
# Required: Your OpenRouter API Key
OPENROUTER_KEY=your_openrouter_api_key_here

# Optional: Path to store the ChromaDB vector database (Defaults to ./chroma_db)
CHROMA_PATH=./chroma_db

# Optional: Path to your processed JSON data file (Defaults to ./ProcessedData/processed_all_no_branches.json)
DATA_FILE=./ProcessedData/processed_all_no_branches.json
```

### 4. Prepare the Data

Ensure your scraped and processed data file (`processed_all_no_branches.json`) is placed in the location specified by the `DATA_FILE` environment variable.

### 5. Run the CLI Assistant

To run the command-line interface (CLI) version of the assistant:

```bash
python LangChainV.py
```

### 6. Run the Streamlit Web Interface

To launch the interactive web application:

```bash
streamlit run streamlitUI.py
```

This will open the WE Smart Assistant in your web browser, providing a user-friendly chat interface with file upload capabilities.

You can run the main script to initialize the database (it will automatically build the ChromaDB if it doesn't exist) and start the CLI chat interface:

```bash
python LangChainV.py
```

---

## 🧠 How It Works (The Pipeline)

1. **User Input**: The user asks a question (e.g., "بكام باقة وي اير؟").

1. **Preprocessing**: The system checks for greetings. If it's a complex query, it strips conversational intent and condenses it using chat history if ambiguous.

1. **Routing**: Regex signals detect "WE Air" and apply a metadata filter (`domain: internet_services`).

1. **Retrieval**: The Ensemble Retriever fetches the top `K` documents using both BM25 (keyword) and ChromaDB (semantic).

1. **Reranking**: The Cross-Encoder reranks the retrieved documents to find the absolute most relevant chunks.

1. **Generation**: The context is injected into a strict, persona-driven prompt, and the LLM generates a precise, localized response.

---

## 📝 License

This project is proprietary and developed for WE Telecom Egypt by Marwan Hatem. All rights reserved.

