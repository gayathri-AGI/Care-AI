from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from dotenv import load_dotenv
import os
import sqlite3
from datetime import datetime

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# -----------------------------------
# FLASK SETUP
# -----------------------------------

app = Flask(__name__)
app.secret_key = "careai_secret_key"

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# -----------------------------------
# DATABASE SETUP
# -----------------------------------

def init_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_message TEXT,
        bot_response TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


# -----------------------------------
# EMBEDDINGS + RETRIEVER
# -----------------------------------

embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)

# ✅ Back to normal (NO strict threshold)
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# -----------------------------------
# LLM
# -----------------------------------

chatModel = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.2
)


# -----------------------------------
# PROMPT
# -----------------------------------

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


# -----------------------------------
# RAG CHAIN
# -----------------------------------

rag_answer_chain = create_stuff_documents_chain(
    chatModel,
    rag_prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    rag_answer_chain
)


# -----------------------------------
# MEMORY STORE
# -----------------------------------

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# -----------------------------------
# ROUTES
# -----------------------------------

# ---------------- LOGIN ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "kalyani" and password == "123":
            session["user"] = username
            return redirect(url_for("chat_page"))
        else:
            return "Invalid username or password"

    return render_template("login.html")


@app.route("/chat")
def chat_page():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("chat.html")


# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        return redirect(url_for("login"))

    return render_template("register.html")


# ---------------- CHAT ----------------
@app.route("/get", methods=["POST"])
def chat():
    try:
        if "user" not in session:
            return jsonify({"answer": "Please login first"})

        msg = request.form.get("msg", "").strip()
        print("User:", msg)

        session_id = session.get("user", "default")

        greetings = ["hi", "hello", "hey", "good morning", "good evening"]

        # ✅ Greeting fix
        if msg.lower() in greetings:
            answer = "Hello 👋 I'm your Medical Assistant. How can I help you today?"
        else:
            response = conversational_rag_chain.invoke(
                {"input": msg},
                config={"configurable": {"session_id": session_id}}
            )

            print("FULL RESPONSE:", response)

            # ✅ Always allow answer (no blocking)
            if isinstance(response, dict):
                answer = response.get("answer") or response.get("output") or str(response)
            else:
                answer = str(response)

        # ✅ Clean encoding
        answer = answer.encode("utf-8", errors="ignore").decode("utf-8").strip()

        print("Bot:", answer)

        # ✅ Save chat
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO chats(session_id, user_message, bot_response, timestamp)
        VALUES (?, ?, ?, ?)
        """, (session_id, msg, answer, str(datetime.now())))

        conn.commit()
        conn.close()

        return jsonify({"answer": answer})

    except Exception as e:
        print("FULL ERROR:", str(e))
        return jsonify({"answer": str(e)})


# -----------------------------------
# VIEW CHAT HISTORY
# -----------------------------------

@app.route("/history")
def get_history():

    session_id = session.get("user", "default")

    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT user_message, bot_response
    FROM chats
    WHERE session_id=?
    ORDER BY id DESC
    LIMIT 20
    """, (session_id,))

    rows = cursor.fetchall()
    conn.close()

    history = [{"user": row[0], "bot": row[1]} for row in rows]

    return jsonify(history)


# -----------------------------------
# CLEAR CHAT
# -----------------------------------

@app.route("/clear")
def clear_history():

    session_id = session.get("user", "default")

    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM chats WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()

    store.pop(session_id, None)

    return "History cleared"


# -----------------------------------
# RUN
# -----------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)