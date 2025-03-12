import os
import psycopg2
from flask import Flask, request, jsonify, render_template
import requests
from together import Together
from dotenv import load_dotenv
from fuzzywuzzy import process
import re

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
        return response.json()["data"]["link"]  # Get the public URL of the image
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

#Initialisation Db
def initialize_database():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Check if the table already exists
        cursor.execute("SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = 'product_table_new');")
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            # Read the .sql file and execute it
            with open("product_table_sample.sql", "r") as sql_file:
                sql_script = sql_file.read()
                cursor.execute(sql_script)
                connection.commit()
                print("Database initialized successfully!")
        else:
            print("Database already initialized.")

        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Error initializing database: {e}")

# Initialize the database when the app starts
initialize_database()

# Function to fetch all medicine names from the database
def fetch_all_medicines():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT medicine FROM product_table_new")
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return [result[0] for result in results]  # PostgreSQL returns tuples, so use index 0

# Function to search for medicine in the database
def search_medicine_in_db(medicine_name):
    first_word = medicine_name.split()[0]  # Take the first word for better matching

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    cursor.execute("SELECT medicine FROM product_table_new WHERE medicine LIKE %s", (f"{first_word}%",))
    results = cursor.fetchall()
    
    cursor.close()
    connection.close()

    return [result['medicine'] for result in results]

def clean_medicine_name(medicine_name):
    """ Remove unwanted suffixes like 'ST ST', numbers, and special characters """
    medicine_name = medicine_name.lower().strip()
    medicine_name = re.sub(r'\b(st|ml|kg|mg|tablet|capsule|syrup)\b', '', medicine_name)  # Remove common suffixes
    medicine_name = re.sub(r'[^a-zA-Z0-9\s]', '', medicine_name)  # Remove special chars
    medicine_name = re.sub(r'\s+', ' ', medicine_name).strip()  # Remove extra spaces
    return medicine_name


# Function to get similar medicine names using fuzzy matching
def get_similar_medicines(medicine_name, all_medicines, limit=3, threshold=50):
    if not medicine_name or not all_medicines:
        return []

    medicine_name = medicine_name.lower()
    all_medicines = [med.lower() for med in all_medicines]

    first_three_chars = medicine_name[:3]  # Take the first 3 letters

    # Filter medicines that start with the same 3 characters
    filtered_medicines = [med for med in all_medicines if med.startswith(first_three_chars)]

    if not filtered_medicines:
        return []  

    # Apply fuzzy matching on filtered results
    matches = process.extract(medicine_name, filtered_medicines, limit=limit)

    # Filter matches based on similarity threshold
    similar_medicines = [match[0] for match in matches if match[1] >= threshold]

    return similar_medicines


# Function to parse medicine name and quantity
def parse_medicine_and_quantity(text):
    # Remove unwanted symbols (*, -, etc.)
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text).strip()
    
    # Extract medicine name and quantity
    match = re.match(r"([a-zA-Z\s]+)\s*(\d+)?", cleaned_text)
    if match:
        medicine_name = match.group(1).strip()
        quantity = match.group(2).strip() if match.group(2) else "1"
        return medicine_name, quantity
    else:
        return cleaned_text, "1"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        # Save the uploaded image locally
        image_file = request.files['image']
        image_path = f"./temp/{image_file.filename}"
        os.makedirs("./temp", exist_ok=True)
        image_file.save(image_path)

        # Upload image to Imgur and get the public URL
        uploaded_image_url = upload_to_imgur(image_path)

        # Send request to Together AI's Vision Model
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all medicine names and quantities from the image. Return the result as a list in the format: {Medicine Name} {Quantity}. Separate each item with a new line."},
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

        # Split the extracted text into individual items
        items = extracted_text.strip().split("\n")

        # Fetch all medicines from the database for fuzzy matching
        all_medicines = fetch_all_medicines()

        results = []
        for item in items:
            medicine_name, quantity = parse_medicine_and_quantity(item)

            # Search for the medicine in the database
            matched_medicines = search_medicine_in_db(medicine_name)
            matched_medicine = matched_medicines[0] if matched_medicines else "No match found"

            # Get similar medicine names if no exact match is found
            suggestions = []
            if matched_medicine == "No match found":
                suggestions = get_similar_medicines(medicine_name, all_medicines)

            # Add to cart if the medicine is found in the database
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
