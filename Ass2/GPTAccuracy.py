import pandas as pd
from openai import OpenAI
import os

# Read the CSV file
df = pd.read_csv('Suicide_Detection.csv')
#get dataset from here https://www.kaggle.com/code/rutujapotdar/suicide-text-classification-nlp?select=Suicide_Detection.csv

# Get the column names
columns = df.columns
OPENAI_API_KEY = "insert here"

# Initialize OpenAI client with API key
client = OpenAI(api_key=OPENAI_API_KEY)

correct_count = 0
total_count = 0

print("\nProcessing predictions:")
for i in range(1000):  # Process first 1000 rows
    text_message = df.iloc[i, 1]  # Get the message from 'text' column (index 1)
    actual_label = df.iloc[i, 2]  # Get the actual label from 'class' column (index 2)
    
    # Call the OpenAI API
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify this message based on if this person needs help from the suicide prevention services because they show signs of suicide. Respond with only 'suicide' or 'non-suicide'."},
            {"role": "user", "content": text_message}
        ]
    )
    
    # Extract the prediction from the response
    chatgpt_label = completion.choices[0].message.content.strip().lower()

    # Compare ChatGPT's response with the actual label
    if chatgpt_label == actual_label.lower():
        correct_count += 1

    total_count += 1
    
    # Print accuracy every 10 samples
    if (i + 1) % 10 == 0:
        current_accuracy = (correct_count / total_count) * 100
        print(f"Samples processed: {total_count}, Current Accuracy: {current_accuracy:.2f}%")

# Print final accuracy
accuracy = (correct_count / total_count) * 100
print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct_count}/{total_count} correct)")

