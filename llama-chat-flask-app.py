from flask import Flask, render_template, request, jsonify
from llama_index.llms.llama_api import LlamaAPI
from llama_index.core.llms import ChatMessage
import os
from datetime import datetime

app = Flask(__name__)

# Securely get the API key from an environment variable
api_key = os.environ.get("LLAMA_API_KEY")

if not api_key:
    raise ValueError("No API key set for LlamaAPI. Please set the LLAMA_API_KEY environment variable.")

llm = LlamaAPI(api_key=api_key)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    messages = [
        ChatMessage(
            role="system", 
            content="You are a professional AI assistant. Provide concise, accurate, and helpful responses."
        ),
        ChatMessage(role="user", content=user_message),
    ]
    resp = llm.chat(messages)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return jsonify({
        'response': resp.message,
        'timestamp': timestamp
    })

if __name__ == '__main__':
    app.run(debug=True)
