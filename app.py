from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

app = Flask(__name__)

# ENV VARIABLES
PINECONE_API_KEY = os.environ.get("pcsk_5okcB_SBiv2hHFw5Bo67XDw74jcufr7QJotbYkwigvqfNGDR9voAtJP2fxiSHJdozCesc")
GEMINI_API_KEY   = os.environ.get("AIzaSyC3l98HfaaOLdLGlIjKM522csNim449LfA")

INDEX_NAME = "aistudymate"

# INIT
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer('all-MiniLM-L6-v2')

genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash")

def ask_question(query):
    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    context = ""
    for match in results['matches']:
        context += match['metadata']['text'] + "\n\n"

    prompt = f"""
You are a helpful teacher.

Answer using the context below.

Context:
{context}

Question:
{query}
"""

    response = gemini.generate_content(prompt)
    return response.text


@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    msg = request.values.get("Body", "")
    answer = ask_question(msg)

    resp = MessagingResponse()
    resp.message(answer)

    return str(resp)


if __name__ == "__main__":
    app.run()