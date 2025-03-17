import os
import psycopg2
from flask import Flask, request, jsonify, render_template
import requests
from together import Together
from dotenv import load_dotenv
from fuzzywuzzy import process
import re
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

def preprocess_medicine_name(medicine_name):
    """Standardize the medicine name for better matching."""
    return re.sub(r'\s+', ' ', medicine_name.strip().lower())

def get_relevant_suggestions(medicine_name, all_medicines, limit=5):
    """Get relevant suggestions for the provided medicine name."""
    medicine_name = preprocess_medicine_name(medicine_name)
    first_char = medicine_name[0].lower() if medicine_name else ""

    # Fuzzy matching
    fuzzy_matches = process.extract(medicine_name, all_medicines, limit=limit * 2)
    suggestions = [match[0] for match in fuzzy_matches if match[1] > 70]  

    # Filter by first character
    filtered_suggestions = [med for med in suggestions if med.lower().startswith(first_char)]

    # Phonetic matching fallback
    if not filtered_suggestions:
        medicine_phonetic = doublemetaphone(medicine_name)[0]
        phonetic_matches = [med for med in all_medicines if doublemetaphone(med)[0] == medicine_phonetic]
        return phonetic_matches[:limit]

    return filtered_suggestions[:limit]

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

        image_file = request.files['image']
        image_path = f"./temp/{image_file.filename}"
        os.makedirs("./temp", exist_ok=True)
        image_file.save(image_path)

        # Upload to Imgur
        uploaded_image_url = upload_to_imgur(image_path)

        # Send to Together AI for OCR
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
            max_tokens=None
        )

        extracted_text = response.choices[0].message.content
        items = extracted_text.strip().split("\n")

        all_medicines = fetch_all_medicines()
        results = []

        for item in items:
            medicine_name = preprocess_medicine_name(item)
            exact_match = next((med for med in all_medicines if med.lower() == medicine_name), None)
            first_word = medicine_name.split()[0] if " " in medicine_name else medicine_name
            first_word_matches = [med for med in all_medicines if med.lower().startswith(first_word.lower())]
            suggestions = get_relevant_suggestions(medicine_name, all_medicines)

            # Priority order list (always 7 elements)
            priority_list = []
            if exact_match:
                priority_list.append(exact_match)
            priority_list.extend(first_word_matches[:7 - len(priority_list)])
            priority_list.extend(suggestions[:7 - len(priority_list)])
            priority_list = priority_list[:7]  # Ensure only 7 items

            results.append({
                "extracted_medicine": medicine_name,
                "exact_match": exact_match or "No exact match",
                "suggestions": suggestions[:5],
                "priority_list": priority_list,
                "quantity": "1"
            })

        return jsonify({"results": results, "cart": cart})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
