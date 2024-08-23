## with reduced rows and columns. Its working perfectly.

import pandas as pd
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reading the CSV file with a specific encoding
csv_file_path = r'data/BrCA Dataset_N5030_lab.csv'
temp_csv_path = r'data/processed_brac_data.csv'

# Try different encodings to read the CSV file
encodings = ['ISO-8859-1', 'cp1252', 'utf-16']
df_csv = None

for encoding in encodings:
    try:
        df_csv = pd.read_csv(csv_file_path, encoding=encoding)
        df_csv.to_csv(temp_csv_path, index=False)  # Save to a new CSV file
        print(f"File successfully processed with encoding {encoding}")
        break
    except Exception as e:
        print(f"Failed to read the file with encoding {encoding}: {e}")

# Function to query the LLaMA 3 endpoint
def query_llama3(inputs, context):
    API_URL = os.getenv('API_URL')
    API_TOKEN = os.getenv('API_TOKEN')
    
    if not API_URL or not API_TOKEN:
        raise ValueError("API_URL or API_TOKEN environment variable is not set.")
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": f"{context}\n\nQuestion: {inputs}"
    }
    
    print(f"Payload being sent: {payload}")  # Debugging output
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Check for HTTP request errors
        response_data = response.json()
        
        if isinstance(response_data, list) and len(response_data) > 0:
            return response_data[0].get('generated_text', 'No valid response from the model.')
        else:
            return "No valid response from the model."
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

# Create a minimal context for testing
def generate_answer(question, context):
    context_str = str(context)[:2000]  # Limit context size for testing
    return query_llama3(question, context_str)

if df_csv is not None:
    # Limit to first 10 columns
    columns_to_include = df_csv.columns[:20]  # Select the first 10 columns
    pp = df_csv[columns_to_include].head(5)  # Adjust the number of records as needed
    print("Data Preview:")
    print(pp.shape)
    context = pp.to_dict(orient='records')

    # Format the context as a string for the payload
    context_str = "\n\n".join([f"Record {i+1}: {str(record)}" for i, record in enumerate(context)])
    
    query = "How many rows are there in the provided dictionary for the data?"
    response = generate_answer(query, context_str)
    print(response)
else:
    print("Failed to process the CSV file with all attempted encodings.")
