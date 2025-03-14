from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import os
import re
import fitz  # PyMuPDF for PDF handling
import google.generativeai as genai
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
run_with_ngrok(app)  # This will enable Colab to serve the Flask app

# Initialize model and NLP tools (same as your Colab code)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
unnecessary_words = {"etc", "e.t.c", "eg", "i.e", "viz"}

# Load the generative model
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Function to extract and clean text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)  # Open PDF
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9.%\s]", "", text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in unnecessary_words]
    return " ".join(words)

# API endpoint for summarizing a PDF document
@app.route('/summarize', methods=['POST'])
def summarize_document():
    file = request.files.get('file')
    if file:
        pdf_text = extract_text_from_pdf(file)
        cleaned_text = clean_text(pdf_text)
        cleaned_text = cleaned_text[:10000]  # Trim for API limits
        summary_prompt = f"Please summarize the following document: {cleaned_text}"
        summary_response = model.generate_content(summary_prompt)
        return jsonify({"summary": summary_response.text})
    return jsonify({"error": "No file uploaded"})

# API endpoint for generating policy
@app.route('/generate-policy', methods=['POST'])
def generate_policy():
    policy_type = request.json.get('policy_type')
    scenario = request.json.get('scenario')
    if policy_type and scenario:
        policy_prompt = f"Generate a {policy_type} for the following scenario: {scenario}"
        policy_response = model.generate_content(policy_prompt)
        return jsonify({"policy": policy_response.text})
    return jsonify({"error": "Invalid input for policy generation"})

# API endpoint for serving HTML frontend
@app.route('/')
def index():
    return render_template("index.html")

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
