from flask import Flask, request
import requests
import os

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

app = Flask(__name__)

# =========================
# ENV VARIABLES
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "aistudymate"

# =========================
# INIT SERVICES
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer('all-MiniLM-L6-v2')

genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash")

# =========================
# CORE FUNCTION (RAG)
# =========================
def ask_question(query):
    try:
        query_embedding = model.encode(query).tolist()

        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        context = ""
        sources = ""

        for match in results['matches']:
            text = match['metadata'].get('text', '')
            page = match['metadata'].get('page_number', 'N/A')
            book = match['metadata'].get('book_name', 'NCERT')

            context += text + "\n\n"
            sources += f"{book} - Page {page}\n"

        prompt = f"""
You are a helpful teacher.

Answer in the SAME LANGUAGE as the question.

Explain in simple terms so students can understand easily.

Use the context below to answer.

Context:
{context}

Question:
{query}
"""

        response = gemini.generate_content(prompt)

        final_answer = response.text + "\n\n📚 Sources:\n" + sources
        return final_answer

    except Exception as e:
        return f"Error: {str(e)}"


# =========================
# TELEGRAM SEND FUNCTION
# =========================
def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text[:4000]  # Telegram limit safety
    }

    requests.post(url, json=payload)


# =========================
# TELEGRAM WEBHOOK
# =========================
@app.route("/", methods=["POST"])
def webhook():
    data = request.get_json()

    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")

        answer = ask_question(text)
        send_message(chat_id, answer)

    return "ok"


# =========================
# HEALTH CHECK (optional)
# =========================
@app.route("/", methods=["GET"])
def home():
    return "AI StudyMate Bot is running!"


# =========================
# RUN FOR RENDER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)