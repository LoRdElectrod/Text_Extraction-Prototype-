import os
import psycopg2
from flask import Flask, request, jsonify, render_template
import requests
from together import Together
from dotenv import load_dotenv
from fuzzywuzzy import process
import re
import jellyfish
from metaphone import doublemetaphone

# Load environment variables
load_dotenv()

# Environment variables
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

# Initialize Together AI client
client = Together(api_key=TOGETHER_API_KEY)

# Flask app
app = Flask(__name__, template_folder="templates")

# Temporary cart storage
cart = []

# ---------------------------- Database Functions ----------------------------

def get_db_connection():
    """Connect to PostgreSQL database."""
    return psycopg2.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )

def fetch_all_medicines():
    """Fetch all medicine names from the database."""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT medicine FROM product_table_new")
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return [result[0] for result in results]
    except Exception as e:
        print(f"Database Error: {e}")
        return []

# ---------------------------- Utility Functions ----------------------------

def upload_to_imgur(image_path):
    """Upload an image to Imgur and return the URL."""
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    with open(image_path, "rb") as image_file:
        response = requests.post("https://api.imgur.com/3/upload", headers=headers, files={"image": image_file})

    if response.status_code == 200:
        return response.json()["data"]["link"]
    else:
        raise Exception(f"Imgur upload failed: {response.json()}")

def clean_extracted_text(text):
    """Remove special characters and unwanted symbols from the extracted text."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).strip()

def parse_medicine_and_quantity(text):
    """Extract medicine name and quantity from text."""
    match = re.match(r"([a-zA-Z\s]+)\s*(\d+)", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text, "1"

def get_internal_patterns(word):
    """Get internal patterns of a word for better matching."""
    return set(re.findall(r'[a-zA-Z]{2,}', word))

def get_relevant_suggestions(medicine_name, all_medicines, limit=5):
    medicine_name = medicine_name.lower()
    internal_patterns = get_internal_patterns(medicine_name)

    # 1. Fuzzy Matching
    fuzzy_matches = process.extract(medicine_name, all_medicines, limit=limit)
    fuzzy_suggestions = [match[0] for match in fuzzy_matches if match[1] > 50]

    # 2. Internal Pattern Matching
    internal_matches = []
    for med in all_medicines:
        if any(pattern in med.lower() for pattern in internal_patterns):
            internal_matches.append(med)

    # 3. Combine all methods
    suggestions = list(set(fuzzy_suggestions + internal_matches))  # Remove duplicates
    return suggestions[:limit]  # Limit to the specified number

# ---------------------------- Flask Routes ----------------------------

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process the uploaded image and extract medicines."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # Save uploaded image
        image_file = request.files['image']
        image_path = f"./temp/{image_file.filename}"
        os.makedirs("./temp", exist_ok=True)
        image_file.save(image_path)

        # Upload image to Imgur
        uploaded_image_url = upload_to_imgur(image_path)

        # Send request to Together AI for OCR
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
            stop=[" <|eot_id|>", "<|eom_id|>"]
        )

        extracted_text = response.choices[0].message.content
        items = extracted_text.strip().split("\n")

        # Fetch all medicine names from DB
        all_medicines = fetch_all_medicines()

        results = []
        for item in items:
            medicine_name, quantity = parse_medicine_and_quantity(clean_extracted_text(item))

            # Match using only the first word if it's multi-word
            first_word = medicine_name.split()[0] if " " in medicine_name else medicine_name

            # Search for exact matches
            matched_medicines = [med for med in all_medicines if med.lower() == medicine_name.lower()]
            matched_medicine = matched_medicines[0] if matched_medicines else "No exact match found"

            # Check for first word match
            first_word_matches = [med for med in all_medicines if med.lower().startswith(first_word.lower())]

            # Get suggestions if no exact match is found
            suggestions = get_relevant_suggestions(medicine_name, all_medicines) if matched_medicine == "No exact match found" else []

            # Add matched item to cart if there's a match
            if matched_medicine != "No exact match found":
                cart.append({"medicine": matched_medicine, "quantity": quantity})

            results.append({
                "extracted_medicine": medicine_name,
                "matched_medicine": matched_medicine,
                "first_word_match": first_word_matches[0] if first_word_matches else "No first word match",
                "suggestions": suggestions,
                "quantity": quantity
            })

        return jsonify({"results": results, "cart": cart})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_cart', methods=['GET'])
def get_cart():
    """Retrieve the cart."""
    return jsonify({"cart": cart})

@app.route('/remove_from_cart/<int:index>', methods=['DELETE'])
def remove_from_cart(index):
    """Remove an item from the cart."""
    try:
        if 0 <= index < len(cart):
            cart.pop(index)
        return jsonify({"cart": cart})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
