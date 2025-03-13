import os
import psycopg2
from flask import Flask, request, jsonify
import requests
from together import Together
from dotenv import load_dotenv
from fuzzywuzzy import process
import re
from rapidfuzz import fuzz
from collections import Counter

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

# Initialize Together AI client
client = Together(api_key=TOGETHER_API_KEY)

app = Flask(__name__)

cart = []

# Function to upload image to Imgur
def upload_to_imgur(image_path):
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    with open(image_path, "rb") as image_file:
        response = requests.post("https://api.imgur.com/3/upload", headers=headers, files={"image": image_file})
    
    if response.status_code == 200:
        return response.json()["data"]["link"]
    else:
        print("Imgur Upload Error:", response.json())  # Debugging
        raise Exception("Imgur upload failed")

# Function to connect to the database
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )

# Fetch all medicine names from DB
def fetch_all_medicines():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT medicine FROM product_table_new")
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return [result[0] for result in results]  # Extract names as list

# Function to clean extracted text
def clean_extracted_text(text):
    text = re.sub(r"^[^a-zA-Z]+", "", text)  # Remove leading special characters
    return text.strip()

# Function to parse medicine name and quantity
def parse_medicine_and_quantity(text):
    match = re.match(r"([a-zA-Z\s]+)\s*(\d+)", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text.strip(), "1"

# Generate N-Grams from a word
def generate_ngrams(word, n=2):
    word = word.lower()
    return [word[i:i+n] for i in range(len(word)-n+1)]

# Function to find best DB match and suggestions
def find_best_match(medicine_name, all_medicines):
    medicine_name = medicine_name.lower()
    
    # Primary DB match using best fuzzy ratio
    best_match, best_score = process.extractOne(medicine_name, all_medicines, scorer=fuzz.ratio)
    
    # Generate suggestions using n-grams + fuzzy matching
    extracted_ngrams = generate_ngrams(medicine_name)
    similarity_scores = []
    
    for med in all_medicines:
        med_ngrams = generate_ngrams(med.lower())
        common_ngrams = len(set(extracted_ngrams) & set(med_ngrams))
        similarity_scores.append((med, common_ngrams))
    
    # Sort by n-gram match score
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Apply fuzzy matching on top 10 n-gram matches
    top_n_gram_matches = [match[0] for match in similarity_scores[:10]]
    fuzzy_matches = process.extract(medicine_name, top_n_gram_matches, limit=5, scorer=fuzz.ratio)
    
    # Merge results with threshold
    suggestions = list({match[0] for match in fuzzy_matches if match[1] > 50})
    
    return best_match if best_score > 60 else "No match found", suggestions

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # Save the uploaded image
        image_file = request.files['image']
        image_path = f"./temp/{image_file.filename}"
        os.makedirs("./temp", exist_ok=True)
        image_file.save(image_path)

        # Upload to Imgur
        uploaded_image_url = upload_to_imgur(image_path)
        if not uploaded_image_url:
            return jsonify({"error": "Imgur upload failed"}), 500

        # Send request to Together AI
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all medicine names and quantities from the image. Format: {Medicine Name} {Quantity}"},
                        {"type": "image_url", "image_url": {"url": uploaded_image_url}}
                    ]
                }
            ],
            max_tokens=None,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"]
        )

        extracted_text = response.choices[0].message.content.strip()
        items = extracted_text.split("\n")

        # Fetch all medicine names from DB
        all_medicines = fetch_all_medicines()

        results = []
        for item in items:
            clean_text = clean_extracted_text(item)
            medicine_name, quantity = parse_medicine_and_quantity(clean_text)

            # Get best DB match and suggestions
            matched_medicine, suggestions = find_best_match(medicine_name, all_medicines)

            # Add to cart if matched
            if matched_medicine != "No match found":
                cart.append({"medicine": matched_medicine, "quantity": quantity})

            results.append({
                "extracted_medicine": medicine_name,
                "matched_medicine": matched_medicine,
                "suggestions": suggestions,
                "quantity": quantity
            })

        return jsonify({"results": results, "cart": cart})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
