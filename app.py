from flask import Flask, request, jsonify, render_template
import os
import requests
import tempfile
from together import Together
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")

# Ensure API keys are set
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY is missing. Check your .env file.")
if not IMGUR_CLIENT_ID:
    raise ValueError("IMGUR_CLIENT_ID is missing. Check your .env file.")

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract-text', methods=['POST'])
def extract_text():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        
        # Use temporary file storage
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image_path = temp_file.name
            image_file.save(image_path)

        try:
            # Upload image to Imgur and get the public URL
            uploaded_image_url = upload_to_imgur(image_path)
        finally:
            os.remove(image_path)  # Cleanup the temp file

        # Send request to Together AI's Vision Model
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the text from the image"},
                        {"type": "image_url", "image_url": {"url": uploaded_image_url}}
                    ]
                }
            ],
            max_tokens=500,  # Limiting tokens to avoid API errors
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1
        )

        extracted_text = response.choices[0].message.content if response.choices else "No text extracted."

        return jsonify({"extracted_text": extracted_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
