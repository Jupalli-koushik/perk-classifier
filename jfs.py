import requests
import json
import csv
import pandas as pd
import time

# URL for the API endpoint
url = 'http://172.172.204.65:8090/tgi/generate'

# Prompt template
prompt_template = '''
Here is the prompt:
"
Task: I need to categorize the following entities based on their industry and market positioning.
  For each entity, please provide:
  Entity name: {entity_name}
  Industry: What industry type does the company belong to?
  Sub Category: What Sub Category of industry does the entity belong to?
  Market Positioning: Is the entity categorized as economy, premium, or luxury?
  Brief reason for classification: In max 10 words
  I want the output in json format and in one file which can be exported
"

give only json format of the answer
'''

# Function to save the response into a CSV file
def saving(response):
    json_string = response['generated_text']
    start_index = json_string.find('```') + 3
    end_index = json_string.rfind('```')
    cleaned_json_string = json_string[start_index:end_index].strip()
    
    # Parse the cleaned JSON string
    try:
        data = json.loads(cleaned_json_string)
        entity_data = data[0]
        
        # Define the CSV file name
        csv_file = 'updated_entity_data.csv'
        
        # Open the CSV file in append mode
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # If the file is empty, write the header
            if file.tell() == 0:
                writer.writerow(["Entity name", "Industry", "Sub Category", "Market Positioning", "Brief reason for classification"])
            
            # Write data as a new row
            writer.writerow([
                entity_data["Entity name"],
                entity_data["Industry"],
                entity_data["Sub Category"],
                entity_data["Market Positioning"],
                entity_data["Brief reason for classification"]
            ])
        
        print(f"Data successfully appended to {csv_file}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

# Function to generate the prompt and process the response
def generate():
    df = pd.read_csv("part3.csv")
    entity_names = df["Entity Name"].tolist()
    for entity_name in entity_names:
        prompt = prompt_template.format(entity_name=entity_name)
        response = get_response_until_success(url, prompt)
        saving(response)

# Function to make the POST request to the API with retry until successful
def get_response_until_success(url, prompt):
    headers = {
        'Content-Type': 'application/json'
    }
    
    output_tokens = 4096 - len(prompt.split()) - 1
    max_tokens = min(output_tokens, 1000)
    
    data = {
        'inputs': prompt,
        'parameters': {
            'max_new_tokens': max_tokens
        }
    }
    
    while True:
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code}: Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying in 5 seconds...")
        # time.sleep(2)  # Wait for 5 seconds before retrying

# Run the generate function
generate()
