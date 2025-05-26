import pandas as pd
import os
import shutil
import time
import numpy as np
import openai
import random
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader, TextLoader

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
client = openai.OpenAI(api_key=openai_api_key)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to load and split CSV data
def load_and_split_data(file_path, test_size=0.01):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Clean the data - remove rows with missing values in key columns
    df = df.dropna(subset=['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'])
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split into train and test
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Data split complete: {len(train_df)} training samples, {len(test_df)} test samples")
    
    return train_df, test_df

# Function to create documents from dataframe
def create_documents_from_df(df):
    documents = []
    for idx, row in df.iterrows():
        # Extract search term from first column
        search_term = row.iloc[0]
        content = f"Search terms report: {search_term}\n: {idx}"
        
        # Create document with metadata
        document = {
            "page_content": content,
            "metadata": {"row": idx}
        }
        documents.append(document)
    
    print(f"Created {len(documents)} documents")
    return documents

# Function to create or load vector database
def create_vector_db(documents, db_name="faiss_index_testing", force_rebuild=False):
    # Check if database exists and we're not forcing a rebuild
    if os.path.exists(db_name) and not force_rebuild:
        print(f"Loading existing database from {db_name}...")
        try:
            db = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
            print(f"Successfully loaded database with {db.index.ntotal} vectors")
            return db
        except Exception as e:
            print(f"Error loading database: {e}")
            print("Will rebuild the database instead")
            if os.path.exists(db_name):
                shutil.rmtree(db_name, ignore_errors=True)
    
    # If we get here, we need to build the database
    if not documents:
        raise ValueError("Documents must be provided to create a new database")
    
    # Remove existing database if it exists
    if os.path.exists(db_name):
        shutil.rmtree(db_name, ignore_errors=True)
        print(f"Removed existing database at {db_name}")
    
    # Create new database with progress indicators
    print(f"Creating new vector database with {len(documents)} documents...")
    
    # Convert documents to LangChain document format
    from langchain_core.documents import Document
    langchain_docs = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in documents]
    
    # Process in batches to show progress
    batch_size = max(1, len(langchain_docs) // 10)  # Show progress for 10 batches
    all_batches = [langchain_docs[i:i+batch_size] for i in range(0, len(langchain_docs), batch_size)]
    
    db = None
    for i, batch in enumerate(all_batches):
        print(f"Processing batch {i+1}/{len(all_batches)}...")
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)
    
    # Save the database
    print("Saving database...")
    db.save_local(db_name)
    print(f"Created and saved database to {db_name}")
    
    return db

# Function to query the database
def query_database(db, query, k=20):
    print(f"Querying database with: '{query}'")
    start_time = time.time()
    results = db.similarity_search(query, k=k)
    end_time = time.time()
    print(f"Query completed in {end_time - start_time:.2f} seconds")
    return results

# Function to extract metrics from search results
def extract_metrics_from_results(results, df):
    formatted_results = []
    
    for doc in results:
        # Initialize variables for key metrics
        search_term = "N/A"
        match_type = "N/A"
        avg_cpm = "N/A"
        impressions = "N/A"
        interactions = "N/A"
        conversions = "N/A"
        
        # Try to get the original row data from metadata
        if 'row' in doc.metadata:
            row_num = doc.metadata['row']
            
            # Get the full row data
            if 0 <= row_num < len(df):
                row_data = df.iloc[row_num]
                
                # Extract the search term from the first column
                if len(row_data) > 0:
                    search_term = row_data.iloc[0]
                
                # Extract match type (second column, index 1)
                if 'Unnamed: 1' in row_data.index and pd.notna(row_data['Unnamed: 1']):
                    match_type = row_data['Unnamed: 1']
                
                # Map the unnamed columns to the correct metrics based on position
                # Avg. CPM is typically in column 6 (index 6)
                if 'Unnamed: 6' in row_data.index and pd.notna(row_data['Unnamed: 6']):
                    avg_cpm = row_data['Unnamed: 6']
                
                # Impressions is typically in column 7 (index 7)
                if 'Unnamed: 7' in row_data.index and pd.notna(row_data['Unnamed: 7']):
                    impressions = row_data['Unnamed: 7']
                
                # Interactions is typically in column 8 (index 8)
                if 'Unnamed: 8' in row_data.index and pd.notna(row_data['Unnamed: 8']):
                    interactions = row_data['Unnamed: 8']
                
                # Conversions is typically in column 14 (index 14)
                if 'Unnamed: 14' in row_data.index and pd.notna(row_data['Unnamed: 14']):
                    conversions = row_data['Unnamed: 14']
        
        # Add to formatted results for LLM
        formatted_results.append({
            "search_term": search_term,
            "match_type": match_type,
            "avg_cpm": avg_cpm,
            "impressions": impressions,
            "interactions": interactions,
            "conversions": conversions
        })
    
    return formatted_results

# Function to predict metrics using OpenAI
def predict_metrics_with_llm(query, similar_results):
    print(f"Predicting metrics for: '{query}'")
    
    # Format the similar results for the prompt
    formatted_examples = ""
    for i, result in enumerate(similar_results):
        formatted_examples += f"Example {i+1}:\n"
        formatted_examples += f"Search Term: {result['search_term']}\n"
        formatted_examples += f"Match Type: {result['match_type']}\n"
        formatted_examples += f"Avg. CPM: {result['avg_cpm']}\n"
        formatted_examples += f"Impressions: {result['impressions']}\n"
        formatted_examples += f"Interactions: {result['interactions']}\n"
        formatted_examples += f"Conversions: {result['conversions']}\n\n"
    
    # Create the prompt
    prompt = f"""
Based on the similar search terms and their metrics below, predict the exact metrics for: "{query}"

{formatted_examples}

Format your response exactly like this:
Predicted Average CPM: [exact number]
Predicted Impressions: [exact number]
Predicted Interactions: [exact number]
Predicted Conversions: [exact number]

Reasoning:
• [brief bullet point]
• [brief bullet point]
• [brief bullet point]

Keep your response concise. No additional text or explanations.
"""

    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using a high reasoning model
            messages=[
                {"role": "system", "content": "You are a digital marketing analyst who provides concise, direct predictions with minimal explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent predictions
            max_tokens=500
        )
        
        # Extract and return the prediction
        prediction = response.choices[0].message.content
        return prediction
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Unable to generate prediction due to an error."

# Function to parse prediction from LLM response
def parse_prediction(prediction_text):
    try:
        lines = prediction_text.strip().split('\n')
        metrics = {}
        
        for line in lines:
            if "Predicted Average CPM:" in line:
                value = line.split(":", 1)[1].strip()
                try:
                    metrics["avg_cpm"] = float(value)
                except ValueError:
                    metrics["avg_cpm"] = None
            elif "Predicted Impressions:" in line:
                value = line.split(":", 1)[1].strip()
                try:
                    metrics["impressions"] = float(value)
                except ValueError:
                    metrics["impressions"] = None
            elif "Predicted Interactions:" in line:
                value = line.split(":", 1)[1].strip()
                try:
                    metrics["interactions"] = float(value)
                except ValueError:
                    metrics["interactions"] = None
            elif "Predicted Conversions:" in line:
                value = line.split(":", 1)[1].strip()
                try:
                    metrics["conversions"] = float(value)
                except ValueError:
                    metrics["conversions"] = None
        
        return metrics
    except Exception as e:
        print(f"Error parsing prediction: {e}")
        return {}

# Function to calculate metrics
def calculate_metrics(actual, predicted):
    metrics = {}
    
    # Calculate RMSE
    if actual is not None and predicted is not None:
        mse = mean_squared_error([actual], [predicted])
        rmse = np.sqrt(mse)
        metrics["rmse"] = rmse
        
        # Calculate percentage deviation
        if actual != 0:
            pct_dev = abs(predicted - actual) / abs(actual) * 100
            metrics["pct_dev"] = pct_dev
        else:
            metrics["pct_dev"] = None
    else:
        metrics["rmse"] = None
        metrics["pct_dev"] = None
    
    return metrics

# Main testing function
def run_test():
    print("Starting RAG testing with 99/1 train-test split...")
    
    # Load and split data
    train_df, test_df = load_and_split_data("Search_terms.csv")
    
    # Check if database already exists
    db_name = "faiss_index_testing"
    if os.path.exists(db_name):
        print(f"Found existing testing database at {db_name}")
        try:
            db = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
            print(f"Successfully loaded database with {db.index.ntotal} vectors")
        except Exception as e:
            print(f"Error loading existing database: {e}")
            print("Creating new database instead...")
            train_documents = create_documents_from_df(train_df)
            db = create_vector_db(train_documents, db_name=db_name, force_rebuild=True)
    else:
        # Create documents from training data
        print("No existing database found. Creating new one...")
        train_documents = create_documents_from_df(train_df)
        db = create_vector_db(train_documents, db_name=db_name, force_rebuild=True)
    
    # Initialize results storage
    results = []
    
    # Track metrics
    all_cpm_rmse = []
    all_cpm_pct_dev = []
    all_impressions_rmse = []
    all_impressions_pct_dev = []
    all_interactions_rmse = []
    all_interactions_pct_dev = []
    
    # Test on ALL test data (the full 1%)
    print(f"Testing on all {len(test_df)} test samples...")
    
    # Run tests
    for i, (idx, row) in enumerate(test_df.iterrows()):
        # Get the search term
        search_term = row.iloc[0]
        
        # Get actual metrics
        actual_avg_cpm = float(row['Unnamed: 6']) if pd.notna(row['Unnamed: 6']) else None
        actual_impressions = float(row['Unnamed: 7']) if pd.notna(row['Unnamed: 7']) else None
        actual_interactions = float(row['Unnamed: 8']) if pd.notna(row['Unnamed: 8']) else None
        
        # Query the database
        query_results = query_database(db, search_term, k=20)
        
        # Extract metrics from results
        formatted_results = extract_metrics_from_results(query_results, train_df)
        
        # Predict metrics
        prediction_text = predict_metrics_with_llm(search_term, formatted_results)
        
        # Parse prediction
        parsed_prediction = parse_prediction(prediction_text)
        
        # Calculate metrics
        if actual_avg_cpm is not None and parsed_prediction.get("avg_cpm") is not None:
            cpm_metrics = calculate_metrics(actual_avg_cpm, parsed_prediction["avg_cpm"])
            all_cpm_rmse.append(cpm_metrics["rmse"])
            if cpm_metrics["pct_dev"] is not None:
                all_cpm_pct_dev.append(cpm_metrics["pct_dev"])
        
        if actual_impressions is not None and parsed_prediction.get("impressions") is not None:
            impressions_metrics = calculate_metrics(actual_impressions, parsed_prediction["impressions"])
            all_impressions_rmse.append(impressions_metrics["rmse"])
            if impressions_metrics["pct_dev"] is not None:
                all_impressions_pct_dev.append(impressions_metrics["pct_dev"])
        
        if actual_interactions is not None and parsed_prediction.get("interactions") is not None:
            interactions_metrics = calculate_metrics(actual_interactions, parsed_prediction["interactions"])
            all_interactions_rmse.append(interactions_metrics["rmse"])
            if interactions_metrics["pct_dev"] is not None:
                all_interactions_pct_dev.append(interactions_metrics["pct_dev"])
        
        # Store result
        result = {
            "search_term": search_term,
            "actual_avg_cpm": actual_avg_cpm,
            "predicted_avg_cpm": parsed_prediction.get("avg_cpm"),
            "actual_impressions": actual_impressions,
            "predicted_impressions": parsed_prediction.get("impressions"),
            "actual_interactions": actual_interactions,
            "predicted_interactions": parsed_prediction.get("interactions")
        }
        results.append(result)
        
        # Print progress every 100 tests
        if (i + 1) % 100 == 0:
            # Calculate current metrics
            current_metrics = calculate_current_metrics(
                all_cpm_rmse, all_cpm_pct_dev,
                all_impressions_rmse, all_impressions_pct_dev,
                all_interactions_rmse, all_interactions_pct_dev
            )
            
            # Print only the metrics (nothing else)
            print(f"Progress: {i+1}/{len(test_df)} samples")
            print(f"Current CPM RMSE: {current_metrics['avg_cpm_rmse']:.2f}")
            print(f"Current CPM MPD: {current_metrics['avg_cpm_pct_dev']:.2f}%")
            print(f"Current Impressions RMSE: {current_metrics['avg_impressions_rmse']:.2f}")
            print(f"Current Impressions MPD: {current_metrics['avg_impressions_pct_dev']:.2f}%")
            print(f"Current Interactions RMSE: {current_metrics['avg_interactions_rmse']:.2f}")
            print(f"Current Interactions MPD: {current_metrics['avg_interactions_pct_dev']:.2f}%")
            print(f"Current Accuracy Score: {current_metrics['accuracy_score']:.2f}%")
            print("-" * 50)
    
    # Calculate final overall metrics
    final_metrics = calculate_current_metrics(
        all_cpm_rmse, all_cpm_pct_dev,
        all_impressions_rmse, all_impressions_pct_dev,
        all_interactions_rmse, all_interactions_pct_dev
    )
    
    print("\n=== FINAL TEST RESULTS ===")
    print(f"Average CPM RMSE: {final_metrics['avg_cpm_rmse']:.2f}")
    print(f"Average CPM Percentage Deviation: {final_metrics['avg_cpm_pct_dev']:.2f}%")
    print(f"Impressions RMSE: {final_metrics['avg_impressions_rmse']:.2f}")
    print(f"Impressions Percentage Deviation: {final_metrics['avg_impressions_pct_dev']:.2f}%")
    print(f"Interactions RMSE: {final_metrics['avg_interactions_rmse']:.2f}")
    print(f"Interactions Percentage Deviation: {final_metrics['avg_interactions_pct_dev']:.2f}%")
    print(f"\nOVERALL ACCURACY SCORE: {final_metrics['accuracy_score']:.2f}%")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("rag_test_results.csv", index=False)
    print("\nTest results saved to 'rag_test_results.csv'")

# Helper function to calculate current metrics
def calculate_current_metrics(cpm_rmse, cpm_pct_dev, imp_rmse, imp_pct_dev, int_rmse, int_pct_dev):
    metrics = {}
    
    # Filter out None values
    valid_cpm_rmse = [x for x in cpm_rmse if x is not None]
    valid_cpm_pct_dev = [x for x in cpm_pct_dev if x is not None]
    valid_impressions_rmse = [x for x in imp_rmse if x is not None]
    valid_impressions_pct_dev = [x for x in imp_pct_dev if x is not None]
    valid_interactions_rmse = [x for x in int_rmse if x is not None]
    valid_interactions_pct_dev = [x for x in int_pct_dev if x is not None]
    
    # Calculate means if data exists
    metrics["avg_cpm_rmse"] = np.mean(valid_cpm_rmse) if valid_cpm_rmse else float('nan')
    metrics["avg_cpm_pct_dev"] = np.mean(valid_cpm_pct_dev) if valid_cpm_pct_dev else float('nan')
    metrics["avg_impressions_rmse"] = np.mean(valid_impressions_rmse) if valid_impressions_rmse else float('nan')
    metrics["avg_impressions_pct_dev"] = np.mean(valid_impressions_pct_dev) if valid_impressions_pct_dev else float('nan')
    metrics["avg_interactions_rmse"] = np.mean(valid_interactions_rmse) if valid_interactions_rmse else float('nan')
    metrics["avg_interactions_pct_dev"] = np.mean(valid_interactions_pct_dev) if valid_interactions_pct_dev else float('nan')
    
    # Calculate overall accuracy score
    all_valid_pct_devs = valid_cpm_pct_dev + valid_impressions_pct_dev + valid_interactions_pct_dev
    if all_valid_pct_devs:
        avg_pct_dev = np.mean(all_valid_pct_devs)
        metrics["accuracy_score"] = max(0, 100 - min(avg_pct_dev, 100))
    else:
        metrics["accuracy_score"] = float('nan')
    
    return metrics

if __name__ == "__main__":
    run_test()
