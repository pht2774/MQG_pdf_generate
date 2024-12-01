from flask import Flask, request, render_template, jsonify
import PyPDF2
import random
from transformers import pipeline

app = Flask(__name__)

# Load the question-generation model
qa_model = pipeline("question-generation", model="valhalla/t5-small-qg-prepend")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to process text into sentences
def process_text(text):
    sentences = text.split(". ")
    return [s.strip() for s in sentences if len(s.split()) > 5]

# Function to generate distractors
def generate_distractors(answer, sentences):
    words = [word.strip(",. ") for sentence in sentences for word in sentence.split()]
    return random.sample([word for word in words if word.lower() != answer.lower()], 3)

# Function to generate MCQs
def generate_mcqs(sentences):
    mcqs = []
    for sentence in sentences:
        try:
            questions = qa_model(sentence)
            for q in questions:
                question = q['question']
                correct_answer = q['answer']
                distractors = generate_distractors(correct_answer, sentences)
                options = list(set(distractors + [correct_answer]))
                random.shuffle(options)
                mcqs.append({
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer
                })
        except Exception as e:
            print(f"Error generating question: {e}")
    return mcqs

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle PDF uploads and generate MCQs
@app.route('/generate', methods=['POST'])
def generate():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    pdf_file = request.files['pdf_file']
    if not pdf_file.filename.endswith('.pdf'):
        return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400

    text = extract_text_from_pdf(pdf_file)
    key_sentences = process_text(text)
    mcqs = generate_mcqs(key_sentences)
    
    return jsonify(mcqs)

if __name__ == '__main__':
    app.run(debug=True)
