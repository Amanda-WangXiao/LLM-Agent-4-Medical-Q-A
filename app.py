# app.py

from flask import Flask, render_template, request, jsonify
from agent import run_agent

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_input = request.json['query']
    response = run_agent(user_input)
    print("Agent response:", response)  # print in console
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)