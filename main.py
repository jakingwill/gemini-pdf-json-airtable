import os
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
from pathlib import Path
import json
from flask import Flask, request, jsonify

# Load Google API key from environment variable (more secure)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
AIRTABLE_WEBHOOK_URL = os.environ.get('AIRTABLE_WEBHOOK_URL')  # Set your Airtable webhook URL here

# Error handling for missing API key or webhook URL
if not GOOGLE_API_KEY:
    raise ValueError("Missing Google API Key. Please set the 'GOOGLE_API_KEY' environment variable.")
if not AIRTABLE_WEBHOOK_URL:
    raise ValueError("Missing Airtable Webhook URL. Please set the 'AIRTABLE_WEBHOOK_URL' environment variable.")

genai.configure(api_key=GOOGLE_API_KEY)

def configure_model():
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    system_instruction = (
        "You are a document entity extraction specialist for a school that gives you assessments. "
        "Given an assessment, your task is to extract the text value of the following entities:\n"
        "{\n \"question\": [\n  {\n    \"question_number\": \"\",\n    \"total_marks\": \"\",\n    "
        "\"question_text\": \"\",\n    \"marking_guide\": \"\"\n  }\n ],\n \"answer\": [\n  {\n    "
        "\"question_number\": \"\",\n    \"student_answer\": \"\"\n  }\n ],\n}\n\n- The JSON schema "
        "must be followed during the extraction.\n- The values must only include text strings found "
        "in the document.\n- Generate null for missing entities.\n- Return the output in JSON format."
    )

    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=system_instruction
    )

def download_pdf(pdf_url, output_path):
    response = requests.get(pdf_url)
    with open(output_path, 'wb') as file:
        file.write(response.content)

def extract_pdf_text(filepath):
    try:
        doc = fitz.open(filepath)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():  # Check if the page text is not empty
                print(f"Extracted text from page {page_num + 1}:")
                print(page_text)
                text += page_text + "\n"  # Add a newline between pages
        return text
    except FileNotFoundError:
        print(f"Error: PDF file not found: {filepath}")
        return ""

app = Flask(__name__)

@app.route('/process-assessment', methods=['POST'])
def process_assessment():
    data = request.json
    record_id = data.get('recordId')
    pdf_url = data.get('pdfUrl')

    if not record_id or not pdf_url:
        return jsonify({"success": False, "error": "Missing recordId or pdfUrl"}), 400

    model = configure_model()

    pdf_path = Path("/tmp/assessment.pdf")  # Temporary path to save the downloaded PDF

    # Download the PDF from the provided URL
    download_pdf(pdf_url, pdf_path)

    # Extract text from the downloaded PDF
    extracted_text = extract_pdf_text(str(pdf_path))
    if not extracted_text:
        return jsonify({"success": False, "error": "No text extracted from the PDF"}), 400

    print("Extracted text from PDF:")
    print(extracted_text)

    user_input = {
        "role": "user",
        "parts": [extracted_text]
    }

    convo = model.start_chat(history=[
        {
            "role": "model",
            "parts": ["[\n  {\n    \"question number\": \"3.1\",\n    \"question\": \"Based on your Term 3 documentary that you submitted, answer the following questions:\\nTitle of Documentary\",\n    \"marks for question\": \"8\",\n    \"student answer\": \"Liverpool and manchester united rivalry\"\n  },\n  {\n    \"question number\": \"3.1.1\",\n    \"question\": \"Provide a definition of B-Roll footage\",\n    \"marks for question\": \"1\",\n    \"student answer\": \"Footage taken from other videos from youtube, tiktok, and other streaming platforms.\"\n  },\n  {\n    \"question number\": \"3.1.2\",\n    \"question\": \"Give TWO clear examples where you used B-Roll footage in your documentary\",\n    \"marks for question\": \"2\",\n    \"student answer\": \"Liverpool lifting the champions league after winning, The footage of the fans\"\n  },\n  {\n    \"question number\": \"3.1.3\",\n    \"question\": \"What was the overall tone of your documentary? (e.g. serious, humorous, critical)\",\n    \"marks for question\": \"2\",\n    \"student answer\": null, \"role\": \"user\"\n  }\n]"]
        }
    ])

    # Only send a message if the extracted text is not empty
    if extracted_text.strip():
        response = convo.send_message(user_input['parts'][0])
        print("Model response:")
        print(response.text)
        # Parse the response as JSON
        response_json = json.loads(response.text)
        print("Parsed JSON response:")
        print(json.dumps(response_json, indent=2))
        # Send the JSON response to Airtable webhook
        webhook_response = requests.post(AIRTABLE_WEBHOOK_URL, json=response_json)
        if webhook_response.status_code == 200:
            print("JSON response sent to Airtable webhook successfully")
            return jsonify({"success": True}), 200
        else:
            print(f"Error sending JSON response to Airtable webhook: {webhook_response.status_code} - {webhook_response.text}")
            return jsonify({"success": False, "error": webhook_response.text}), 500
    else:
        return jsonify({"success": False, "error": "Extracted text is empty"}), 400

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
