from flask import Flask, render_template, request, redirect, url_for, session
from llama_index.core import SimpleDirectoryReader
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import json
import os
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = './uploads'
app.secret_key = 'supersecretkey'

# Initialize embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to chunk text
def chunk_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        pdf = PdfReader(f)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        return text

# Function to process uploaded PDF
def process_uploaded_file(filepath):
    global embedding_store, chunk_store  
    text = extract_text_from_pdf(filepath)
    chunks = chunk_text(text)
    chunk_embeddings = embedder.encode(chunks)
    embedding_store = np.array(chunk_embeddings)
    chunk_store = chunks

# Function to find relevant chunks based on user query
def find_relevant_chunks(query, top_k=5):
    query_embedding = embedder.encode([query])[0]
    similarities = np.dot(embedding_store, query_embedding)
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    relevant_chunks = [chunk_store[i] for i in top_k_indices]
    return relevant_chunks

# Function to prepare combined query
def prepare_combined_query(user_query, query_length):
    if "summary" in user_query.lower():
        combined_query = "Provide a summary of the following text:\n\n"
        combined_query += ' '.join(chunk_store)  # Use all chunks for summarization
        combined_query += f"\n\nPlease provide a response that is {query_length}."
    else:
        relevant_chunks = find_relevant_chunks(user_query)
        combined_query = user_query + "\n\n" + ' '.join(relevant_chunks) + f"\n\nPlease provide a response that is {query_length}."
    return combined_query

# Function to send combined query to Mixtral LLM
def send_to_mixtral(combined_query):
    try:
        url = "https://mixtral.k8s-gosha.atlas.illinois.edu/completion"
        myobj = {
            "prompt": "<s>[INST]"+combined_query+"[/INST]",
            "n_predict": -1
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(url, data=json.dumps(myobj), headers=headers,
                                 auth=('atlasaiteam', 'jx@U2WS8BGSqwu'), timeout=1000)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filename)
            # Process the uploaded file
            process_uploaded_file(filename)
            return redirect(url_for('query'))
    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    if 'query_history' not in session:
        session['query_history'] = []

    if request.method == 'POST':
        if 'query' in request.form:
            user_query = request.form['query']
            query_length = request.form['length']
            combined_query = prepare_combined_query(user_query, query_length)
            response = send_to_mixtral(combined_query)

            # Extract the desired content from the response
            if response and 'content' in response:
                content = response['content']
            else:
                content = 'No relevant information found.'

            # Save query and response to history
            session['query_history'].append({'query': user_query, 'response': content})
            session.modified = True

        if 'clear_history' in request.form:
            session.pop('query_history', None)
            return redirect(url_for('query'))

    return render_template('query.html', query_history=session['query_history'])

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('query_history', None)
    return redirect(url_for('query'))

if __name__ == '__main__':
    app.run(debug=True)