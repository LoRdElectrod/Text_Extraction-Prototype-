import os
import psycopg2
from flask import Flask, request, jsonify, render_template
import requests
from together import Together
from dotenv import load_dotenv
from fuzzywuzzy import process
import re
from rapidfuzz import fuzz  # Better for string similarity
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

app = Flask(__name__, template_folder="templates")

cart = []

# Function to upload image to Imgur
def upload_to_imgur(image_path):
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    with open(image_path, "rb") as image_file:
        response = requests.post("https://api.imgur.com/3/upload", headers=headers, files={"image": image_file})
    
    if response.status_code == 200:
        return response.json()["data"]["link"]
    else:
        raise Exception(f"Imgur upload failed: {response.json()}")

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

# Function to clean extracted text (remove garbage symbols)
def clean_extracted_text(text):
    text = re.sub(r"^[^a-zA-Z]+", "", text)  # Remove leading special characters
    return text.strip()

# Function to parse medicine name and quantity
def parse_medicine_and_quantity(text):
    match = re.match(r"([a-zA-Z\s]+)\s*(\d+)", text)
    if match:
        medicine_name = match.group(1).strip()
        quantity = match.group(2).strip()
        return medicine_name, quantity
    else:
        return text, "1"

# Generate N-Grams from a word
def generate_ngrams(word, n=2):
    """Generate character n-grams for better similarity matching"""
    word = word.lower()
    return [word[i:i+n] for i in range(len(word)-n+1)]

# Function to get relevant medicine suggestions based on similarity
def get_relevant_suggestions(medicine_name, all_medicines, limit=5):
    medicine_name = medicine_name.lower()
    
    # Use fuzzy matching to get the best suggestions
    fuzzy_matches = process.extract(medicine_name, all_medicines, limit=limit, scorer=fuzz.ratio)

    # Filter matches with a high similarity score
    suggestions = [match[0] for match in fuzzy_matches if match[1] > 50]

    return list(set(suggestions))  # Remove duplicates

# Function to find the best database match
def get_best_db_match(medicine_name, all_medicines):
    matches = process.extract(medicine_name, all_medicines, limit=1, scorer=fuzz.ratio)
    if matches and matches[0][1] > 70:  # Ensure high confidence match
        return matches[0][0]
    return "No match found"

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

        extracted_text = response.choices[0].message.content
        items = extracted_text.strip().split("\n")

        # Fetch all medicine names from DB
        all_medicines = fetch_all_medicines()

        results = []
        for item in items:
            clean_text = clean_extracted_text(item)  # Remove unwanted symbols
            medicine_name, quantity = parse_medicine_and_quantity(clean_text)

            # Get best match from DB
            matched_medicine = get_best_db_match(medicine_name, all_medicines)

            # Get suggestions based on similarity
            suggestions = get_relevant_suggestions(medicine_name, all_medicines)

            # Add to cart if a match is found
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
