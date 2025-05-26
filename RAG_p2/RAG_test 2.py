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

# Function to load and split CSV data into evergreen and non-evergreen
def load_and_split_data(file_path, test_size=0.001):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Clean the data - remove rows with missing values in key columns
    df = df.dropna(subset=['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'])
    
    # Split into evergreen and non-evergreen based on column 3 (index 3)
    # Check if the string "[Evergreen]" appears in the column
    evergreen_mask = df['Unnamed: 3'].str.contains('\[Evergreen\]', na=False)
    evergreen_df = df[evergreen_mask]
    non_evergreen_df = df[~evergreen_mask]
    
    print(f"Split data into {len(evergreen_df)} evergreen rows and {len(non_evergreen_df)} non-evergreen rows")
    
    # Shuffle the dataframes
    evergreen_df = evergreen_df.sample(frac=1, random_state=42).reset_index(drop=True)
    non_evergreen_df = non_evergreen_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split points (using 0.1% for test)
    evergreen_split_idx = int(len(evergreen_df) * (1 - test_size))
    non_evergreen_split_idx = int(len(non_evergreen_df) * (1 - test_size))
    
    # Split into train and test
    evergreen_train_df = evergreen_df.iloc[:evergreen_split_idx]
    evergreen_test_df = evergreen_df.iloc[evergreen_split_idx:]
    
    non_evergreen_train_df = non_evergreen_df.iloc[:non_evergreen_split_idx]
    non_evergreen_test_df = non_evergreen_df.iloc[non_evergreen_split_idx:]
    
    print(f"Evergreen data split: {len(evergreen_train_df)} training, {len(evergreen_test_df)} test samples")
    print(f"Non-evergreen data split: {len(non_evergreen_train_df)} training, {len(non_evergreen_test_df)} test samples")
    
    # Combine test sets for evaluation
    test_df = pd.concat([evergreen_test_df, non_evergreen_test_df])
    
    return evergreen_train_df, non_evergreen_train_df, test_df, evergreen_mask

# Function to create documents from dataframe
def create_documents_from_df(df, is_evergreen):
    documents = []
    category = "evergreen" if is_evergreen else "non_evergreen"
    
    for idx, row in df.iterrows():
        # Extract search term from first column
        search_term = row.iloc[0]
        
        # Extract match type (second column, index 1)
        match_type = row['Unnamed: 1'] if pd.notna(row['Unnamed: 1']) else "Unknown Match Type"
        
        # Include match type in the content
        content = f"Search terms report: {search_term}\nMatch Type: {match_type}\nCategory: {category}\nRow: {idx}"
        
        # Create document with metadata
        document = {
            "page_content": content,
            "metadata": {
                "row": idx,
                "category": category,
                "match_type": match_type
            }
        }
        documents.append(document)
    
    print(f"Created {len(documents)} {category} documents")
    return documents

# Function to create or load vector database
def create_vector_db(documents, db_name, force_rebuild=False):
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

# Function to query the appropriate database based on whether the query is evergreen or not
def query_database(evergreen_db, non_evergreen_db, query, is_evergreen, match_type=None, k=20):
    db_to_use = evergreen_db if is_evergreen else non_evergreen_db
    db_type = "evergreen" if is_evergreen else "non-evergreen"
    
    # Enhance query with match type if provided
    enhanced_query = query
    if match_type:
        enhanced_query = f"Search terms report: {query}\nMatch Type: {match_type}"
    
    print(f"Querying {db_type} database with: '{enhanced_query}'")
    start_time = time.time()
    results = db_to_use.similarity_search(enhanced_query, k=k)
    end_time = time.time()
    print(f"Query completed in {end_time - start_time:.2f} seconds")
    return results

# Function to extract metrics from search results
def extract_metrics_from_results(results, evergreen_train_df, non_evergreen_train_df):
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
        if 'row' in doc.metadata and 'category' in doc.metadata:
            row_num = doc.metadata['row']
            category = doc.metadata['category']
            
            # Select the appropriate dataframe based on category
            df_to_use = evergreen_train_df if category == "evergreen" else non_evergreen_train_df
            
            # Get the full row data
            if 0 <= row_num < len(df_to_use):
                row_data = df_to_use.iloc[row_num]
                
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
def predict_metrics_with_llm(query, similar_results, match_type=None):
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
    
    # Add match type information to the prompt if provided
    match_type_info = ""
    if match_type:
        match_type_info = f"\nThe user has selected match type: {match_type}. Consider this when making your prediction."
    
    # Create the prompt
    prompt = f"""
Based on the similar search terms and their metrics below, predict the exact metrics for: "{query}"{match_type_info}

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
    print("Starting RAG testing with 99.9/0.1 train-test split and separate evergreen/non-evergreen databases...")
    
    # Load and split data
    evergreen_train_df, non_evergreen_train_df, test_df, evergreen_mask = load_and_split_data("Search_terms.csv", test_size=0.001)
    
    # Create documents from training data for each category
    evergreen_documents = create_documents_from_df(evergreen_train_df, is_evergreen=True)
    non_evergreen_documents = create_documents_from_df(non_evergreen_train_df, is_evergreen=False)
    
    # Create or load vector databases
    evergreen_db_name = "faiss_index_evergreen"
    non_evergreen_db_name = "faiss_index_non_evergreen"
    
    # Check if databases already exist
    if os.path.exists(evergreen_db_name) and os.path.exists(non_evergreen_db_name):
        print(f"Found existing databases")
        try:
            evergreen_db = FAISS.load_local(evergreen_db_name, embeddings, allow_dangerous_deserialization=True)
            non_evergreen_db = FAISS.load_local(non_evergreen_db_name, embeddings, allow_dangerous_deserialization=True)
            print(f"Successfully loaded evergreen database with {evergreen_db.index.ntotal} vectors")
            print(f"Successfully loaded non-evergreen database with {non_evergreen_db.index.ntotal} vectors")
        except Exception as e:
            print(f"Error loading existing databases: {e}")
            print("Creating new databases instead...")
            evergreen_db = create_vector_db(evergreen_documents, db_name=evergreen_db_name, force_rebuild=True)
            non_evergreen_db = create_vector_db(non_evergreen_documents, db_name=non_evergreen_db_name, force_rebuild=True)
    else:
        # Create new databases
        print("No existing databases found. Creating new ones...")
        evergreen_db = create_vector_db(evergreen_documents, db_name=evergreen_db_name, force_rebuild=True)
        non_evergreen_db = create_vector_db(non_evergreen_documents, db_name=non_evergreen_db_name, force_rebuild=True)
    
    # Initialize results storage
    results = []
    
    # Track metrics
    all_cpm_rmse = []
    all_cpm_pct_dev = []
    all_impressions_rmse = []
    all_impressions_pct_dev = []
    all_interactions_rmse = []
    all_interactions_pct_dev = []
    
    # Test on ALL test data
    print(f"Testing on all {len(test_df)} test samples...")
    
    # Run tests
    for i, (idx, row) in enumerate(test_df.iterrows()):
        # Get the search term
        search_term = row.iloc[0]
        
        # Determine if this is an evergreen search term
        is_evergreen = evergreen_mask.loc[idx] if idx in evergreen_mask.index else False
        db_type = "evergreen" if is_evergreen else "non-evergreen"
        
        # Get actual metrics
        actual_avg_cpm = float(row['Unnamed: 6']) if pd.notna(row['Unnamed: 6']) else None
        actual_impressions = float(row['Unnamed: 7']) if pd.notna(row['Unnamed: 7']) else None
        actual_interactions = float(row['Unnamed: 8']) if pd.notna(row['Unnamed: 8']) else None
        
        # Query the appropriate database
        query_results = query_database(evergreen_db, non_evergreen_db, search_term, is_evergreen, k=20)
        
        # Extract metrics from results
        formatted_results = extract_metrics_from_results(query_results, evergreen_train_df, non_evergreen_train_df)
        
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
            if actual_interactions != 0:
                interactions_metrics = calculate_metrics(actual_interactions, parsed_prediction["interactions"])
                all_interactions_rmse.append(interactions_metrics["rmse"])
                interactions_pct_dev = abs(parsed_prediction["interactions"] - actual_interactions) / abs(actual_interactions) * 100
                all_interactions_pct_dev.append(interactions_pct_dev)
            else:
                # Handle zero actual interactions
                if parsed_prediction["interactions"] == 0:
                    # If both are zero, 0% deviation
                    all_interactions_pct_dev.append(0)
                else:
                    # If prediction is non-zero but actual is zero, use a large value or special handling
                    all_interactions_pct_dev.append(100)  # 100% error or another approach
        
        # Store result
        result = {
            "search_term": search_term,
            "is_evergreen": is_evergreen,
            "actual_avg_cpm": actual_avg_cpm,
            "predicted_avg_cpm": parsed_prediction.get("avg_cpm"),
            "actual_impressions": actual_impressions,
            "predicted_impressions": parsed_prediction.get("impressions"),
            "actual_interactions": actual_interactions,
            "predicted_interactions": parsed_prediction.get("interactions")
        }
        results.append(result)
        
        # Print progress every 10 tests (since we have a smaller test set now)
        if (i + 1) % 10 == 0:
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
    
    # Save accuracy score to a text file
    with open("accuracy_score.txt", "w") as f:
        f.write(f"OVERALL ACCURACY SCORE: {final_metrics['accuracy_score']:.2f}%\n\n")
        f.write("=== DETAILED METRICS ===\n")
        f.write(f"Average CPM RMSE: {final_metrics['avg_cpm_rmse']:.2f}\n")
        f.write(f"Average CPM Percentage Deviation: {final_metrics['avg_cpm_pct_dev']:.2f}%\n")
        f.write(f"Impressions RMSE: {final_metrics['avg_impressions_rmse']:.2f}\n")
        f.write(f"Impressions Percentage Deviation: {final_metrics['avg_impressions_pct_dev']:.2f}%\n")
        f.write(f"Interactions RMSE: {final_metrics['avg_interactions_rmse']:.2f}\n")
        f.write(f"Interactions Percentage Deviation: {final_metrics['avg_interactions_pct_dev']:.2f}%\n")
        f.write(f"\nTest completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nAccuracy score saved to 'accuracy_score.txt'")
    
    # Calculate metrics by category (evergreen vs non-evergreen)
    evergreen_results = [r for r in results if r["is_evergreen"]]
    non_evergreen_results = [r for r in results if not r["is_evergreen"]]
    
    print("\n=== EVERGREEN VS NON-EVERGREEN COMPARISON ===")
    print(f"Evergreen test samples: {len(evergreen_results)}")
    print(f"Non-evergreen test samples: {len(non_evergreen_results)}")
    
    # Calculate metrics for evergreen
    if evergreen_results:
        evergreen_metrics = calculate_category_metrics(evergreen_results)
        print("\nEVERGREEN METRICS:")
        print(f"Evergreen CPM Percentage Deviation: {evergreen_metrics['avg_cpm_pct_dev']:.2f}%")
        print(f"Evergreen Impressions Percentage Deviation: {evergreen_metrics['avg_impressions_pct_dev']:.2f}%")
        print(f"Evergreen Interactions Percentage Deviation: {evergreen_metrics['avg_interactions_pct_dev']:.2f}%")
        print(f"Evergreen Accuracy Score: {evergreen_metrics['accuracy_score']:.2f}%")
        
        # Add evergreen metrics to the text file
        with open("accuracy_score.txt", "a") as f:
            f.write("\n\n=== EVERGREEN METRICS ===\n")
            f.write(f"Evergreen test samples: {len(evergreen_results)}\n")
            f.write(f"Evergreen CPM Percentage Deviation: {evergreen_metrics['avg_cpm_pct_dev']:.2f}%\n")
            f.write(f"Evergreen Impressions Percentage Deviation: {evergreen_metrics['avg_impressions_pct_dev']:.2f}%\n")
            f.write(f"Evergreen Interactions Percentage Deviation: {evergreen_metrics['avg_interactions_pct_dev']:.2f}%\n")
            f.write(f"Evergreen Accuracy Score: {evergreen_metrics['accuracy_score']:.2f}%\n")
    
    # Calculate metrics for non-evergreen
    if non_evergreen_results:
        non_evergreen_metrics = calculate_category_metrics(non_evergreen_results)
        print("\nNON-EVERGREEN METRICS:")
        print(f"Non-Evergreen CPM Percentage Deviation: {non_evergreen_metrics['avg_cpm_pct_dev']:.2f}%")
        print(f"Non-Evergreen Impressions Percentage Deviation: {non_evergreen_metrics['avg_impressions_pct_dev']:.2f}%")
        print(f"Non-Evergreen Interactions Percentage Deviation: {non_evergreen_metrics['avg_interactions_pct_dev']:.2f}%")
        print(f"Non-Evergreen Accuracy Score: {non_evergreen_metrics['accuracy_score']:.2f}%")
        
        # Add non-evergreen metrics to the text file
        with open("accuracy_score.txt", "a") as f:
            f.write("\n=== NON-EVERGREEN METRICS ===\n")
            f.write(f"Non-evergreen test samples: {len(non_evergreen_results)}\n")
            f.write(f"Non-Evergreen CPM Percentage Deviation: {non_evergreen_metrics['avg_cpm_pct_dev']:.2f}%\n")
            f.write(f"Non-Evergreen Impressions Percentage Deviation: {non_evergreen_metrics['avg_impressions_pct_dev']:.2f}%\n")
            f.write(f"Non-Evergreen Interactions Percentage Deviation: {non_evergreen_metrics['avg_interactions_pct_dev']:.2f}%\n")
            f.write(f"Non-Evergreen Accuracy Score: {non_evergreen_metrics['accuracy_score']:.2f}%\n")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("rag_test_results.csv", index=False)
    print("\nTest results saved to 'rag_test_results.csv'")

# Helper function to calculate metrics for a specific category
def calculate_category_metrics(category_results):
    cpm_pct_devs = []
    impressions_pct_devs = []
    interactions_pct_devs = []
    
    for result in category_results:
        # CPM percentage deviation
        if result["actual_avg_cpm"] is not None and result["predicted_avg_cpm"] is not None and result["actual_avg_cpm"] != 0:
            cpm_pct_dev = abs(result["predicted_avg_cpm"] - result["actual_avg_cpm"]) / abs(result["actual_avg_cpm"]) * 100
            cpm_pct_devs.append(cpm_pct_dev)
        
        # Impressions percentage deviation
        if result["actual_impressions"] is not None and result["predicted_impressions"] is not None and result["actual_impressions"] != 0:
            impressions_pct_dev = abs(result["predicted_impressions"] - result["actual_impressions"]) / abs(result["actual_impressions"]) * 100
            impressions_pct_devs.append(impressions_pct_dev)
        
        # Interactions percentage deviation
        if result["actual_interactions"] is not None and result["predicted_interactions"] is not None and result["actual_interactions"] != 0:
            interactions_pct_dev = abs(result["predicted_interactions"] - result["actual_interactions"]) / abs(result["actual_interactions"]) * 100
            interactions_pct_devs.append(interactions_pct_dev)
    
    metrics = {}
    metrics["avg_cpm_pct_dev"] = np.mean(cpm_pct_devs) if cpm_pct_devs else float('nan')
    metrics["avg_impressions_pct_dev"] = np.mean(impressions_pct_devs) if impressions_pct_devs else float('nan')
    metrics["avg_interactions_pct_dev"] = np.mean(interactions_pct_devs) if interactions_pct_devs else float('nan')
    
    # Calculate overall accuracy score
    all_pct_devs = cpm_pct_devs + impressions_pct_devs + interactions_pct_devs
    metrics["accuracy_score"] = max(0, 100 - min(np.mean(all_pct_devs), 100)) if all_pct_devs else float('nan')
    
    return metrics

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

# Function to calculate accuracy metrics for evergreen and non-evergreen data
def calculate_accuracy_metrics(test_size=0.001):
    """
    Calculate accuracy metrics for the RAG model on both evergreen and non-evergreen data.
    
    Args:
        test_size: The proportion of data to use for testing (default: 0.001 or 0.1%)
        
    Returns:
        A dictionary containing accuracy metrics for both categories
    """
    print("Calculating accuracy metrics for RAG model...")
    
    # Load and split data
    evergreen_train_df, non_evergreen_train_df, test_df, evergreen_mask = load_and_split_data("Search_terms.csv", test_size=test_size)
    
    # Create documents from training data for each category
    evergreen_documents = create_documents_from_df(evergreen_train_df, is_evergreen=True)
    non_evergreen_documents = create_documents_from_df(non_evergreen_train_df, is_evergreen=False)
    
    # Create or load vector databases
    evergreen_db_name = "faiss_index_evergreen"
    non_evergreen_db_name = "faiss_index_non_evergreen"
    
    # Check if databases already exist
    if os.path.exists(evergreen_db_name) and os.path.exists(non_evergreen_db_name):
        print(f"Found existing databases")
        try:
            evergreen_db = FAISS.load_local(evergreen_db_name, embeddings, allow_dangerous_deserialization=True)
            non_evergreen_db = FAISS.load_local(non_evergreen_db_name, embeddings, allow_dangerous_deserialization=True)
            print(f"Successfully loaded evergreen database with {evergreen_db.index.ntotal} vectors")
            print(f"Successfully loaded non-evergreen database with {non_evergreen_db.index.ntotal} vectors")
        except Exception as e:
            print(f"Error loading existing databases: {e}")
            print("Creating new databases instead...")
            evergreen_db = create_vector_db(evergreen_documents, db_name=evergreen_db_name, force_rebuild=True)
            non_evergreen_db = create_vector_db(non_evergreen_documents, db_name=non_evergreen_db_name, force_rebuild=True)
    else:
        # Create new databases
        print("No existing databases found. Creating new ones...")
        evergreen_db = create_vector_db(evergreen_documents, db_name=evergreen_db_name, force_rebuild=True)
        non_evergreen_db = create_vector_db(non_evergreen_documents, db_name=non_evergreen_db_name, force_rebuild=True)
    
    # Initialize results storage
    results = []
    
    # Track metrics separately for evergreen and non-evergreen
    evergreen_cpm_pct_dev = []
    evergreen_impressions_pct_dev = []
    evergreen_interactions_pct_dev = []
    evergreen_cpm_rmse = []
    evergreen_impressions_rmse = []
    evergreen_interactions_rmse = []
    
    non_evergreen_cpm_pct_dev = []
    non_evergreen_impressions_pct_dev = []
    non_evergreen_interactions_pct_dev = []
    non_evergreen_cpm_rmse = []
    non_evergreen_impressions_rmse = []
    non_evergreen_interactions_rmse = []
    
    # Test on ALL test data
    print(f"Testing on all {len(test_df)} test samples...")
    
    # Run tests
    for i, (idx, row) in enumerate(test_df.iterrows()):
        # Get the search term
        search_term = row.iloc[0]
        
        # Determine if this is an evergreen search term
        is_evergreen = evergreen_mask.loc[idx] if idx in evergreen_mask.index else False
        db_type = "evergreen" if is_evergreen else "non-evergreen"
        
        # Get match type
        match_type = row['Unnamed: 1'] if pd.notna(row['Unnamed: 1']) else None
        
        # Get actual metrics
        actual_avg_cpm = float(row['Unnamed: 6']) if pd.notna(row['Unnamed: 6']) else None
        actual_impressions = float(row['Unnamed: 7']) if pd.notna(row['Unnamed: 7']) else None
        actual_interactions = float(row['Unnamed: 8']) if pd.notna(row['Unnamed: 8']) else None
        
        # Query the appropriate database
        query_results = query_database(evergreen_db, non_evergreen_db, search_term, is_evergreen, match_type=match_type, k=20)
        
        # Extract metrics from results
        formatted_results = extract_metrics_from_results(query_results, evergreen_train_df, non_evergreen_train_df)
        
        # Predict metrics with match type
        prediction_text = predict_metrics_with_llm(search_term, formatted_results, match_type)
        
        # Parse prediction
        parsed_prediction = parse_prediction(prediction_text)
        
        # Calculate percentage deviations and store by category
        if is_evergreen:
            # CPM percentage deviation and RMSE
            if actual_avg_cpm is not None and parsed_prediction.get("avg_cpm") is not None:
                # Calculate RMSE
                mse = mean_squared_error([actual_avg_cpm], [parsed_prediction["avg_cpm"]])
                rmse = np.sqrt(mse)
                evergreen_cpm_rmse.append(rmse)
                
                # Calculate percentage deviation
                if actual_avg_cpm != 0:
                    cpm_pct_dev = abs(parsed_prediction["avg_cpm"] - actual_avg_cpm) / abs(actual_avg_cpm) * 100
                    evergreen_cpm_pct_dev.append(cpm_pct_dev)
            
            # Impressions percentage deviation and RMSE
            if actual_impressions is not None and parsed_prediction.get("impressions") is not None:
                # Calculate RMSE
                mse = mean_squared_error([actual_impressions], [parsed_prediction["impressions"]])
                rmse = np.sqrt(mse)
                evergreen_impressions_rmse.append(rmse)
                
                # Calculate percentage deviation
                if actual_impressions != 0:
                    impressions_pct_dev = abs(parsed_prediction["impressions"] - actual_impressions) / abs(actual_impressions) * 100
                    evergreen_impressions_pct_dev.append(impressions_pct_dev)
            
            # Interactions percentage deviation and RMSE
            if actual_interactions is not None and parsed_prediction.get("interactions") is not None:
                # Calculate RMSE
                mse = mean_squared_error([actual_interactions], [parsed_prediction["interactions"]])
                rmse = np.sqrt(mse)
                evergreen_interactions_rmse.append(rmse)
                
                # Calculate percentage deviation
                if actual_interactions != 0:
                    interactions_pct_dev = abs(parsed_prediction["interactions"] - actual_interactions) / abs(actual_interactions) * 100
                    evergreen_interactions_pct_dev.append(interactions_pct_dev)
                else:
                    # Handle zero actual interactions
                    if parsed_prediction["interactions"] == 0:
                        # If both are zero, 0% deviation
                        evergreen_interactions_pct_dev.append(0)
                    else:
                        # If prediction is non-zero but actual is zero, use a large value or special handling
                        evergreen_interactions_pct_dev.append(100)  # 100% error or another approach
        else:
            # CPM percentage deviation and RMSE
            if actual_avg_cpm is not None and parsed_prediction.get("avg_cpm") is not None:
                # Calculate RMSE
                mse = mean_squared_error([actual_avg_cpm], [parsed_prediction["avg_cpm"]])
                rmse = np.sqrt(mse)
                non_evergreen_cpm_rmse.append(rmse)
                
                # Calculate percentage deviation
                if actual_avg_cpm != 0:
                    cpm_pct_dev = abs(parsed_prediction["avg_cpm"] - actual_avg_cpm) / abs(actual_avg_cpm) * 100
                    non_evergreen_cpm_pct_dev.append(cpm_pct_dev)
            
            # Impressions percentage deviation and RMSE
            if actual_impressions is not None and parsed_prediction.get("impressions") is not None:
                # Calculate RMSE
                mse = mean_squared_error([actual_impressions], [parsed_prediction["impressions"]])
                rmse = np.sqrt(mse)
                non_evergreen_impressions_rmse.append(rmse)
                
                # Calculate percentage deviation
                if actual_impressions != 0:
                    impressions_pct_dev = abs(parsed_prediction["impressions"] - actual_impressions) / abs(actual_impressions) * 100
                    non_evergreen_impressions_pct_dev.append(impressions_pct_dev)
            
            # Interactions percentage deviation and RMSE
            if actual_interactions is not None and parsed_prediction.get("interactions") is not None:
                # Calculate RMSE
                mse = mean_squared_error([actual_interactions], [parsed_prediction["interactions"]])
                rmse = np.sqrt(mse)
                non_evergreen_interactions_rmse.append(rmse)
                
                # Calculate percentage deviation
                if actual_interactions != 0:
                    interactions_pct_dev = abs(parsed_prediction["interactions"] - actual_interactions) / abs(actual_interactions) * 100
                    non_evergreen_interactions_pct_dev.append(interactions_pct_dev)
                else:
                    # Handle zero actual interactions
                    if parsed_prediction["interactions"] == 0:
                        # If both are zero, 0% deviation
                        non_evergreen_interactions_pct_dev.append(0)
                    else:
                        # If prediction is non-zero but actual is zero, use a large value or special handling
                        non_evergreen_interactions_pct_dev.append(100)  # 100% error or another approach
        
        # Store result
        result = {
            "search_term": search_term,
            "is_evergreen": is_evergreen,
            "match_type": match_type,
            "actual_avg_cpm": actual_avg_cpm,
            "predicted_avg_cpm": parsed_prediction.get("avg_cpm"),
            "actual_impressions": actual_impressions,
            "predicted_impressions": parsed_prediction.get("impressions"),
            "actual_interactions": actual_interactions,
            "predicted_interactions": parsed_prediction.get("interactions")
        }
        results.append(result)
        
        # Print progress every 10 tests
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(test_df)} samples")
    
    # Calculate metrics for evergreen
    evergreen_metrics = {}
    if evergreen_cpm_pct_dev:
        evergreen_metrics["avg_cpm_pct_dev"] = np.mean(evergreen_cpm_pct_dev)
    else:
        evergreen_metrics["avg_cpm_pct_dev"] = float('nan')
    
    if evergreen_impressions_pct_dev:
        evergreen_metrics["avg_impressions_pct_dev"] = np.mean(evergreen_impressions_pct_dev)
    else:
        evergreen_metrics["avg_impressions_pct_dev"] = float('nan')
    
    if evergreen_interactions_pct_dev:
        evergreen_metrics["avg_interactions_pct_dev"] = np.mean(evergreen_interactions_pct_dev)
    else:
        evergreen_metrics["avg_interactions_pct_dev"] = float('nan')
    
    # Add RMSE metrics for evergreen
    if evergreen_cpm_rmse:
        evergreen_metrics["avg_cpm_rmse"] = np.mean(evergreen_cpm_rmse)
    else:
        evergreen_metrics["avg_cpm_rmse"] = float('nan')
    
    if evergreen_impressions_rmse:
        evergreen_metrics["avg_impressions_rmse"] = np.mean(evergreen_impressions_rmse)
    else:
        evergreen_metrics["avg_impressions_rmse"] = float('nan')
    
    if evergreen_interactions_rmse:
        evergreen_metrics["avg_interactions_rmse"] = np.mean(evergreen_interactions_rmse)
    else:
        evergreen_metrics["avg_interactions_rmse"] = float('nan')
    
    # Calculate overall accuracy score for evergreen
    all_evergreen_pct_devs = evergreen_cpm_pct_dev + evergreen_impressions_pct_dev + evergreen_interactions_pct_dev
    if all_evergreen_pct_devs:
        avg_pct_dev = np.mean(all_evergreen_pct_devs)
        evergreen_metrics["accuracy_score"] = max(0, 100 - min(avg_pct_dev, 100))
    else:
        evergreen_metrics["accuracy_score"] = float('nan')
    
    # Calculate metrics for non-evergreen
    non_evergreen_metrics = {}
    if non_evergreen_cpm_pct_dev:
        non_evergreen_metrics["avg_cpm_pct_dev"] = np.mean(non_evergreen_cpm_pct_dev)
    else:
        non_evergreen_metrics["avg_cpm_pct_dev"] = float('nan')
    
    if non_evergreen_impressions_pct_dev:
        non_evergreen_metrics["avg_impressions_pct_dev"] = np.mean(non_evergreen_impressions_pct_dev)
    else:
        non_evergreen_metrics["avg_impressions_pct_dev"] = float('nan')
    
    if non_evergreen_interactions_pct_dev:
        non_evergreen_metrics["avg_interactions_pct_dev"] = np.mean(non_evergreen_interactions_pct_dev)
    else:
        non_evergreen_metrics["avg_interactions_pct_dev"] = float('nan')
    
    # Add RMSE metrics for non-evergreen
    if non_evergreen_cpm_rmse:
        non_evergreen_metrics["avg_cpm_rmse"] = np.mean(non_evergreen_cpm_rmse)
    else:
        non_evergreen_metrics["avg_cpm_rmse"] = float('nan')
    
    if non_evergreen_impressions_rmse:
        non_evergreen_metrics["avg_impressions_rmse"] = np.mean(non_evergreen_impressions_rmse)
    else:
        non_evergreen_metrics["avg_impressions_rmse"] = float('nan')
    
    if non_evergreen_interactions_rmse:
        non_evergreen_metrics["avg_interactions_rmse"] = np.mean(non_evergreen_interactions_rmse)
    else:
        non_evergreen_metrics["avg_interactions_rmse"] = float('nan')
    
    # Calculate overall accuracy score for non-evergreen
    all_non_evergreen_pct_devs = non_evergreen_cpm_pct_dev + non_evergreen_impressions_pct_dev + non_evergreen_interactions_pct_dev
    if all_non_evergreen_pct_devs:
        avg_pct_dev = np.mean(all_non_evergreen_pct_devs)
        non_evergreen_metrics["accuracy_score"] = max(0, 100 - min(avg_pct_dev, 100))
    else:
        non_evergreen_metrics["accuracy_score"] = float('nan')
    
    # Print results
    print("\n=== ACCURACY METRICS ===")
    print(f"Evergreen test samples: {len(evergreen_cpm_pct_dev + evergreen_impressions_pct_dev + evergreen_interactions_pct_dev) // 3}")
    print(f"Non-evergreen test samples: {len(non_evergreen_cpm_pct_dev + non_evergreen_impressions_pct_dev + non_evergreen_interactions_pct_dev) // 3}")
    
    print("\nEVERGREEN METRICS:")
    print(f"Evergreen CPM RMSE: {evergreen_metrics['avg_cpm_rmse']:.2f}")
    print(f"Evergreen CPM Percentage Deviation: {evergreen_metrics['avg_cpm_pct_dev']:.2f}%")
    print(f"Evergreen Impressions RMSE: {evergreen_metrics['avg_impressions_rmse']:.2f}")
    print(f"Evergreen Impressions Percentage Deviation: {evergreen_metrics['avg_impressions_pct_dev']:.2f}%")
    print(f"Evergreen Interactions RMSE: {evergreen_metrics['avg_interactions_rmse']:.2f}")
    print(f"Evergreen Interactions Percentage Deviation: {evergreen_metrics['avg_interactions_pct_dev']:.2f}%")
    print(f"Evergreen Accuracy Score: {evergreen_metrics['accuracy_score']:.2f}%")
    
    print("\nNON-EVERGREEN METRICS:")
    print(f"Non-Evergreen CPM RMSE: {non_evergreen_metrics['avg_cpm_rmse']:.2f}")
    print(f"Non-Evergreen CPM Percentage Deviation: {non_evergreen_metrics['avg_cpm_pct_dev']:.2f}%")
    print(f"Non-Evergreen Impressions RMSE: {non_evergreen_metrics['avg_impressions_rmse']:.2f}")
    print(f"Non-Evergreen Impressions Percentage Deviation: {non_evergreen_metrics['avg_impressions_pct_dev']:.2f}%")
    print(f"Non-Evergreen Interactions RMSE: {non_evergreen_metrics['avg_interactions_rmse']:.2f}")
    print(f"Non-Evergreen Interactions Percentage Deviation: {non_evergreen_metrics['avg_interactions_pct_dev']:.2f}%")
    print(f"Non-Evergreen Accuracy Score: {non_evergreen_metrics['accuracy_score']:.2f}%")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("rag_accuracy_results.csv", index=False)
    print("\nAccuracy test results saved to 'rag_accuracy_results.csv'")
    
    # Return metrics
    return {
        "evergreen": evergreen_metrics,
        "non_evergreen": non_evergreen_metrics
    }

# Interactive search function with LLM prediction
def interactive_search_with_prediction(evergreen_db, non_evergreen_db):
    while True:
        query = input("\nEnter a search term (or 'quit' to exit): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        # Ask if this is an evergreen search term
        while True:
            is_evergreen_input = input("Is this an evergreen search term? (yes/no): ").lower()
            if is_evergreen_input in ['yes', 'y']:
                is_evergreen = True
                break
            elif is_evergreen_input in ['no', 'n']:
                is_evergreen = False
                break
            else:
                print("Please enter 'yes' or 'no'")
        
        # Ask for match type
        print("\nSelect match type:")
        print("1. Broad Match")
        print("2. Exact Match")
        print("3. Phrase Match")
        print("4. Phrase Match (close variant)")
        
        while True:
            match_type_input = input("Enter your choice (1-4): ")
            if match_type_input == '1':
                match_type = "Broad Match"
                break
            elif match_type_input == '2':
                match_type = "Exact Match"
                break
            elif match_type_input == '3':
                match_type = "Phrase Match"
                break
            elif match_type_input == '4':
                match_type = "Phrase Match (close variant)"
                break
            else:
                print("Please enter a number between 1 and 4")
        
        # Perform search to get similar terms from the appropriate database
        results = query_database(evergreen_db, non_evergreen_db, query, is_evergreen, match_type=match_type, k=20)
        
        # Load the appropriate training dataframe
        evergreen_train_df, non_evergreen_train_df, _, _ = load_and_split_data("Search_terms.csv", test_size=0.001)
        
        # Extract metrics from results
        formatted_results = extract_metrics_from_results(results, evergreen_train_df, non_evergreen_train_df)
        
        # Display all retrieved results
        print(f"\n===== ALL RETRIEVED SEARCH RESULTS ({len(formatted_results)} documents) =====")
        for i, result in enumerate(formatted_results):
            print(f"\n--- Result {i+1} ---")
            print(f"Search Term: {result['search_term']}")
            print(f"Match Type: {result['match_type']}")
            print(f"Avg. CPM: {result['avg_cpm']}")
            print(f"Impressions: {result['impressions']}")
            print(f"Interactions: {result['interactions']}")
            print(f"Conversions: {result['conversions']}")
        
        # Use LLM to predict metrics with match type
        prediction = predict_metrics_with_llm(query, formatted_results, match_type)
        
        # Display the prediction
        print("\n🔮 PREDICTED METRICS FOR YOUR SEARCH TERM:")
        print("=" * 50)
        print(prediction)
        print("=" * 50)

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Testing and Interactive Search')
    parser.add_argument('--mode', type=str, choices=['test', 'interactive', 'accuracy'], 
                        default='interactive', help='Mode to run the script in')
    parser.add_argument('--test_size', type=float, default=0.001, 
                        help='Proportion of data to use for testing (default: 0.001)')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Run the full test
        run_test()
    elif args.mode == 'accuracy':
        # Calculate accuracy metrics
        calculate_accuracy_metrics(test_size=args.test_size)
    else:  # interactive mode
        print("Starting interactive search mode...")
        
        # Load and split data
        evergreen_train_df, non_evergreen_train_df, _, _ = load_and_split_data("Search_terms.csv", test_size=args.test_size)
        
        # Create documents from training data for each category
        evergreen_documents = create_documents_from_df(evergreen_train_df, is_evergreen=True)
        non_evergreen_documents = create_documents_from_df(non_evergreen_train_df, is_evergreen=False)
        
        # Create or load vector databases
        evergreen_db_name = "faiss_index_evergreen"
        non_evergreen_db_name = "faiss_index_non_evergreen"
        
        # Check if databases already exist
        if os.path.exists(evergreen_db_name) and os.path.exists(non_evergreen_db_name):
            print(f"Found existing databases")
            try:
                evergreen_db = FAISS.load_local(evergreen_db_name, embeddings, allow_dangerous_deserialization=True)
                non_evergreen_db = FAISS.load_local(non_evergreen_db_name, embeddings, allow_dangerous_deserialization=True)
                print(f"Successfully loaded evergreen database with {evergreen_db.index.ntotal} vectors")
                print(f"Successfully loaded non-evergreen database with {non_evergreen_db.index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading existing databases: {e}")
                print("Creating new databases instead...")
                evergreen_db = create_vector_db(evergreen_documents, db_name=evergreen_db_name, force_rebuild=True)
                non_evergreen_db = create_vector_db(non_evergreen_documents, db_name=non_evergreen_db_name, force_rebuild=True)
        else:
            # Create new databases
            print("No existing databases found. Creating new ones...")
            evergreen_db = create_vector_db(evergreen_documents, db_name=evergreen_db_name, force_rebuild=True)
            non_evergreen_db = create_vector_db(non_evergreen_documents, db_name=non_evergreen_db_name, force_rebuild=True)
        
        # Start interactive search
        print("\n=== Starting Interactive Search with AI Prediction ===")
        print("Now you can search for terms and get predicted metrics based on similar terms")
        interactive_search_with_prediction(evergreen_db, non_evergreen_db)
