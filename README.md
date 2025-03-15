# Medicine OCR and Cart Management System

This project is a **Medicine OCR and Cart Management System** that allows users to upload images of prescriptions, extract medicine names and quantities using AI, and manage a cart for matched medicines. The system uses **Together AI's LLaMA 3.2 Vision Model** for text extraction and a **PostgreSQL database** for medicine matching.

---

## Features

1. **Image Upload**: Users can upload images of prescriptions containing medicine names and quantities.
2. **Text Extraction**: The system extracts medicine names and quantities using AI.
3. **Database Matching**: Extracted medicine names are matched against a database for exact or similar matches.
4. **Cart Management**: Users can add matched medicines to a cart and adjust quantities.
5. **Suggestions**: Provides suggestions for medicines if no exact match is found.
6. **Quantity and Power Separation**: Differentiates between medicine quantity and power (e.g., `500mg`, `1%`).
7. **Phonetic Matching**: Improves suggestion accuracy using phonetic algorithms like Double Metaphone.

---

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript, Tailwind CSS
- **AI Model**: Together AI's LLaMA 3.2 Vision Model
- **Database**: PostgreSQL
- **Image Hosting**: Imgur
- **Fuzzy Matching**: FuzzyWuzzy
- **Phonetic Matching**: Double Metaphone

---

## Setup Instructions

### Prerequisites

1. **Python 3.8+**: Install Python from [python.org](https://www.python.org/).
2. **PostgreSQL**: Install PostgreSQL from [postgresql.org](https://www.postgresql.org/).
3. **Together AI API Key**: Sign up at [Together AI](https://www.together.xyz/) to get an API key.
4. **Imgur Client ID**: Sign up at [Imgur](https://imgur.com/) to get a Client ID.

### Install Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/medicine-ocr-cart-system.git
   cd medicine-ocr-cart-system

2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt

### Environment Variables
## Create a .env file in the root directory and add the following variables:
  TOGETHER_API_KEY=your_together_api_key
  IMGUR_CLIENT_ID=your_imgur_client_id
  DB_HOST=your_database_host
  DB_USER=your_database_user
  DB_PASSWORD=your_database_password
  DB_NAME=your_database_name
  DB_PORT=your_database_port

### Database Setup and Run the Application

### Usage
- **Upload Prescription:** 
- **View Results:**
- **Add to Cart:**
- **View Cart:**

### Acknowledgments
- **Together AI:** For providing the LLaMA 3.2 Vision Model. 
- **Imgur:** For image hosting.
- **Flask:** For the backend framework.
- **Tailwind CSS:**  For frontend styling.
