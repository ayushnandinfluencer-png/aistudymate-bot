from flask import Flask, request
import requests
import os
import json

from pinecone import Pinecone
import google.generativeai as genai

app = Flask(__name__)

# =========================
# ENV VARIABLES
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")   # 🔥 ADD THIS IN RENDER

INDEX_NAME = "aistudymate"

# =========================
# INIT SERVICES
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash")

# =========================
# 🔥 HUGGINGFACE EMBEDDING
# =========================
def get_embedding(text):
    API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    response = requests.post(API_URL, headers=headers, json={"inputs": text})

    if response.status_code != 200:
        return None

    embedding = response.json()

    # Flatten output (important)
    if isinstance(embedding[0], list):
        return embedding[0]

    return embedding

# =========================
# LANGUAGE DETECTION
# =========================
def detect_language_and_intent(query):
    prompt = f"""
Detect:
1. Language of the query
2. If user wants response in another language

Query:
{query}

Respond ONLY in JSON:
{{
"input_language": "...",
"output_language": "...",
"clean_query": "..."
}}
"""
    try:
        response = gemini.generate_content(prompt)
        return json.loads(response.text)
    except:
        return {
            "input_language": "unknown",
            "output_language": "same",
            "clean_query": query
        }

# =========================
# CORE FUNCTION
# =========================
def ask_question(query):
    if not query.strip():
        return "⚠️ Please ask a valid question."

    try:
        lang_data = detect_language_and_intent(query)
        clean_query = lang_data.get("clean_query", query)

        # 🔥 REAL EMBEDDING (FIXED)
        query_embedding = get_embedding(clean_query)

        if not query_embedding:
            return "⚠️ Embedding failed. Try again."

        # 🔍 Pinecone Search
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        if not results.get("matches"):
            return "❌ No relevant data found."

        context = ""
        sources = ""

        for match in results["matches"]:
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            page = metadata.get("page_number", "N/A")
            book = metadata.get("book_name", "NCERT")

            context += text[:500] + "\n\n"
            sources += f"{book} - Page {page}\n"

        prompt = f"""
You are a highly intelligent teacher AI.

Rules:
- Answer in simple student-friendly language
- Answer ONLY from given context
- Do NOT hallucinate

Context:
{context}

Question:
{clean_query}
"""

        response = gemini.generate_content(prompt)

        if not response or not response.text:
            return "⚠️ AI failed. Try again."

        answer = response.text.strip()

        return f"""{answer}

📚 Sources:
{sources}
"""

    except Exception as e:
        return f"❌ Error: {str(e)}"

# =========================
# TELEGRAM SEND
# =========================
def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text[:4000]
    }

    try:
        requests.post(url, json=payload)
    except:
        pass

# =========================
# WEBHOOK
# =========================
@app.route("/", methods=["POST"])
def webhook():
    data = request.get_json()

    try:
        if "message" in data:
            chat_id = data["message"]["chat"]["id"]
            text = data["message"].get("text", "").strip()

            print("User:", text)

            answer = ask_question(text)
            send_message(chat_id, answer)

    except Exception as e:
        print("Webhook Error:", e)

    return "ok"

# =========================
# HEALTH CHECK
# =========================
@app.route("/", methods=["GET"])
def home():
    return "StudyMate AI is running 🚀"

# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)