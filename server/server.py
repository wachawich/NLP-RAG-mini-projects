from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
import os

from dotenv import load_dotenv


from cohere import Client as CohereClient
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

from search import rag_search
from QA import ask

load_dotenv("/opt/.env")
# load_dotenv()

pc = Pinecone(api_key=os.getenv("PINE_CONE_API_KEY"))
index = pc.Index("wiki-embeddings")

co = CohereClient(os.getenv("COHERE_API_KEY"))

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

sentence_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")

app = Flask(__name__)
CORS(app)


@app.route('/api/qa', methods=['POST'])
def process_message():
    """
    QA RAG Search API
    ---
    tags:
      - QA
    description: รับข้อความจากผู้ใช้และประมวลผลด้วย RAG + LLM จากนั้นส่งคำตอบกลับ
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            message:
              type: string
              example: "อธิบายเรื่อง solar forecasting ให้หน่อย"
    responses:
      200:
        description: คำตอบจากโมเดล
        schema:
          type: object
          properties:
            response:
              type: string
              example: "นี่คือคำตอบที่ RAG หาให้..."
    """
    
    data = request.get_json()
    message = data.get('message', '')
    
    result = rag_search(
        groq_client,
        sentence_model,
        index,
        co,
        question=message,
        ns_top_k=4,
        per_namespace=25,
        final_k=10
    )
    
    res = ask(groq_client, result, message)
    
    return jsonify({'response': res})


if __name__ == '__main__':
    app.run(debug=True)