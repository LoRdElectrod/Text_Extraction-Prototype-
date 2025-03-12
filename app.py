import os
import psycopg2
from flask import Flask, request, jsonify, render_template
import requests
from together import Together
from dotenv import load_dotenv
from fuzzywuzzy import process
import re
from difflib import SequenceMatcher

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
    return [result[0] for result in results]

def upload_to_imgur(image_path):
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    with open(image_path, "rb") as image_file:
        image_data = {"image": image_file.read()}
        response = requests.post("https://api.imgur.com/3/upload", headers=headers, files=image_data)
    
    if response.status_code == 200:
        return response.json()["data"]["link"]
    else:
        raise Exception(f"Failed to upload image to Imgur: {response.text}")

# Function to clean extracted text
def clean_extracted_text(text):
    text = re.sub(r"^[^a-zA-Z]+", "", text)  # Remove leading special characters
    return text.strip()

# Function to parse medicine name and quantity
def parse_medicine_and_quantity(text):
    match = re.match(r"([a-zA-Z\s]+)\s*(\d+mg|\d+)", text, re.IGNORECASE)
    if match:
        medicine_name = match.group(1).strip()
        quantity = match.group(2).strip()
        return medicine_name, quantity
    else:
        return text, "1"

# Function to calculate substring similarity
def get_substring_similarity(str1, str2):
    matcher = SequenceMatcher(None, str1.lower(), str2.lower())
    return matcher.ratio()

# Function to find medicine suggestions based on internal pattern matching
def get_relevant_suggestions(medicine_name, all_medicines, quantity="", limit=5):
    medicine_name = medicine_name.lower()
    
    # Step 1: Filter by quantity (if available)
    filtered_medicines = [med for med in all_medicines if quantity.lower() in med.lower()] if quantity else all_medicines

    # Step 2: Apply internal pattern matching using substrings
    scores = []
    for med in filtered_medicines:
        score = get_substring_similarity(medicine_name, med)
        scores.append((med, score))
    
    # Step 3: Sort by best match
    scores.sort(key=lambda x: x[1], reverse=True)

    # Step 4: Return top suggestions
    return [s[0] for s in scores[:limit] if s[1] > 0.4]  # Only keep meaningful matches

@app.route('/')
def index():
    return render_template('index.html')

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

            # Match using only the first word if it's multi-word
            first_word = medicine_name.split()[0] if " " in medicine_name else medicine_name

            # Find relevant suggestions based on internal patterns
            suggestions = get_relevant_suggestions(first_word, all_medicines, quantity)

            # Auto-add best match to cart if high confidence
            matched_medicine = suggestions[0] if suggestions else "No match found"
            if matched_medicine != "No match found":
                cart.append({"medicine": matched_medicine, "quantity": quantity})

            results.append({
                "extracted_medicine": medicine_name,
                "matched_medicine": matched_medicine,
                "suggestions": suggestions,
                "quantity": quantity
            })

        return jsonify({
            "results": results,
            "cart": cart
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_cart', methods=['GET'])
def get_cart():
    return jsonify({"cart": cart})

@app.route('/remove_from_cart/<int:index>', methods=['DELETE'])
def remove_from_cart(index):
    try:
        if 0 <= index < len(cart):
            cart.pop(index)
        return jsonify({"cart": cart})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
