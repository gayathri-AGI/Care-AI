# 🩺 CareAI - AI-Powered Medical Chatbot using LLMs + RAG

**CareAI** is an intelligent, document-aware medical chatbot built using **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)**. It uses vector embeddings to search relevant content from **The Gale Encyclopedia of Medicine (PDF)** and generates accurate, context-driven answers to health-related queries.

---

## 🛠️ Tech Stack

* **Python** 🐍
* **LLaMA 3.2** 🧠 – Large Language Model for answer generation
* **LangChain** 🔗 – Orchestration of retrieval and generation
* **Pinecone** 🌲 – Vector database for similarity search
* **Flask** 🌐 – Lightweight backend for chatbot API
* **PyMuPDF (fitz)** 📄 – PDF parsing and chunking
* **sentence-transformers** 💬 – Text embedding generation
* **HTML/CSS/JS** 🎨 – Simple and responsive frontend interface

---

## ✨ Features

* 📄 Uses *The Gale Encyclopedia of Medicine* as a trusted knowledge base
* 🔎 RAG-powered responses: retrieves relevant context before answering
* 🤖 Uses **LLaMA 3.2** for domain-specific response generation
* 🧠 Fast and accurate **semantic search** with Pinecone
* 💬 Clean web-based chat interface via Flask
* 📂 Modular codebase for easy customization and scaling

---

## 🔍 How It Works

1. **PDF Parsing**: The PDF of *The Gale Encyclopedia of Medicine* is parsed using `PyMuPDF`, and text is chunked intelligently.
2. **Embedding Generation**: Each chunk is converted into high-dimensional vectors using `sentence-transformers`.
3. **Vector Storage**: The embeddings are stored in **Pinecone** for efficient semantic retrieval.
4. **Query Input**: User enters a medical question in the web interface.
5. **RAG Pipeline**: LangChain fetches relevant chunks using Pinecone and passes them to **LLaMA 3.2** to generate an informed answer.
6. **Response Display**: The chatbot responds via the frontend with text output.

---

## 🗂️ Project Structure

```
CareAI-MedicalBot/
│
├── data/                     # Gale Encyclopedia of Medicine PDF
├── embeddings/               # Saved chunk embeddings (optional cache)
├── app.py                    # Flask backend application
├── ingest.py                 # PDF reading and vector creation
├── chat.py                   # LangChain + LLaMA response pipeline
├── pinecone_config.py        # Pinecone setup and API configuration
├── static/                   # CSS and JavaScript for frontend
├── templates/                # HTML template for chat UI
├── requirements.txt          # Python dependency file
└── README.md                 # Project documentation
```

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Main Python Libraries:

* `flask`
* `langchain`
* `pinecone-client`
* `sentence-transformers`
* `llama-cpp-python`
* `PyMuPDF` (`fitz`)
* `dotenv`

---

## 📚 Knowledge Source

> This project uses the PDF version of **The Gale Encyclopedia of Medicine**, a comprehensive and authoritative medical reference covering common conditions, treatments, and procedures.

---

## 🏥 Use Cases

* 💬 **Medical Query Assistant** for patients or learners
* 🏥 Helpdesk Chatbot for hospitals and health websites
* 📚 Educational Aid for medical students and interns
* 💡 Research tool for semantic exploration of medical knowledge
* 🔍 Prototype for domain-specific LLM-powered search engines

---

## 🚀 Feature Improvements

* 🧠 Add **multi-document support** with document tagging
* 🗣️ Integrate **speech-to-text** for voice-based interactions
* 🌐 Deploy using **cloud platforms** (Heroku, Railway, etc.)
* 🔐 Add **user authentication** and history tracking
* 🌍 Incorporate **multi-language support** for wider accessibility
* 📈 Build analytics dashboard to track usage and trends

---
#



# 🎥 Demo
https://github.com/user-attachments/assets/1334c4bb-d561-4011-a9f0-d1dc66e6dae0


---

# HOW TO RUN?
### STEPS:

Clone the repository

```bash
https://github.com/mmannat12/CareAI-MedicalBot.git
```

### STEP 1: CREATE CONDA ENVIRONMENT AFTER OPENING THE REPOSITORY

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

### STEP 2 - INSTALL ALL THE REQUIREMENTS
```bash
 pip install -r requirements.txt
 ```
