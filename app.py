import os
import psycopg2
from flask import Flask, request, jsonify, render_template
import requests
from together import Together
from dotenv import load_dotenv
from fuzzywuzzy import process

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
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT medicine FROM product_table_new WHERE medicine ILIKE %s", (f"%{medicine_name}%",))
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return [result[0] for result in results]  # PostgreSQL returns tuples, so use index 0

# Function to get similar medicine names using fuzzy matching
def get_similar_medicines(medicine_name, all_medicines, limit=5):
    # Use fuzzywuzzy to find the closest matches
    matches = process.extract(medicine_name, all_medicines, limit=limit)
    return [match[0] for match in matches if match[1] > 50]  # Only return matches with a score > 50

# Function to parse medicine name and quantity
def parse_medicine_and_quantity(text):
    # Use regex to extract medicine name and quantity
    # Example: "Coevein 15" -> ("Coevein", "15")
    match = re.match(r"([a-zA-Z\s]+)\s*(\d+)", text)
    if match:
        medicine_name = match.group(1).strip()  # Extract medicine name
        quantity = match.group(2).strip()  # Extract quantity
        return medicine_name, quantity
    else:
        return text, "N/A"  # Fallback if regex doesn't match

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
                        {"type": "text", "text": "Extract only the medicine name and quantity from the image. The given image will be in the format like, {Medicine name}{x quantity},Return the result in the format: {Medicine Name} {Quantity}. Do not include any additional text or explanations."},
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

        # Parse the extracted text to get medicine name and quantity
        medicine_name, quantity = parse_medicine_and_quantity(extracted_text)

        # Fetch all medicines from the database for fuzzy matching
        all_medicines = fetch_all_medicines()

        # Search for the medicine in the database
        matched_medicines = search_medicine_in_db(medicine_name)
        matched_medicine = matched_medicines[0] if matched_medicines else "No match found"

        # Get similar medicine names if no exact match is found
        suggestions = []
        if matched_medicine == "No match found":
            suggestions = get_similar_medicines(medicine_name, all_medicines)

        return jsonify({
            "extracted_medicine": medicine_name,
            "matched_medicine": matched_medicine,
            "suggestions": suggestions,
            "quantity": quantity
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
