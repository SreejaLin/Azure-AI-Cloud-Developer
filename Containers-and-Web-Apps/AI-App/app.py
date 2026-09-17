import os
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

app = Flask(__name__)

# Read configuration from environment variables
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_API_URL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_MODEL_NAME")

client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    response = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=8192,
    )

    return jsonify({
        "model": response.model,
        "reply": response.choices[0].message.content
    })


@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "healthy"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)