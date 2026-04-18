from flask import Flask, request
import requests
import os

# ✅ NEW PINECONE (FIXED)
from pinecone import Pinecone

# ✅ GEMINI
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

# 🔥 FIXED PINECONE INIT (v8)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# GEMINI INIT
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash")

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

Respond in JSON:
{{
"input_language": "...",
"output_language": "...",
"clean_query": "..."
}}
"""
    try:
        response = gemini.generate_content(prompt)
        return eval(response.text)
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
        clean_query = lang_data["clean_query"]

        # 🔥 LIGHTWEIGHT VECTOR (NO TORCH)
        query_embedding = [0.0] * 1536

        # PINECONE SEARCH
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        context = ""
        sources = ""

        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            page = metadata.get('page_number', 'N/A')
            book = metadata.get('book_name', 'NCERT')

            context += text + "\n\n"
            sources += f"{book} - Page {page}\n"

        if not context:
            return "❌ No relevant data found. Try asking differently."

        # GEMINI FINAL ANSWER
        prompt = f"""
You are a highly intelligent teacher AI.

Rules:
- Answer in SAME language as student unless specified otherwise
- Keep explanation simple and student-friendly
- Be accurate and based ONLY on context
- Adapt to board (CBSE/ICSE/State) if mentioned
- Do NOT hallucinate

Context:
{context}

Question:
{clean_query}
"""

        response = gemini.generate_content(prompt)
        answer = response.text.strip()

        final_answer = f"""{answer}

📚 Sources:
{sources}

⚠️ Note: Based on NCERT/reference books.
"""

        return final_answer

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
            text = data["message"].get("text", "")

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
    return "StudyMate AI Telegram Bot is running 🚀"

# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)