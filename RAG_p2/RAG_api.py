import os
import time
import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error
from typing import List, Optional, Dict, Any

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
client = openai.OpenAI(api_key=openai_api_key)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FastAPI app
app = FastAPI(title="RAG Prediction API", 
              description="API for predicting metrics based on search terms using RAG")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define request models
class SearchRequest(BaseModel):
    query: str
    is_evergreen: bool
    match_type: str
    k: int = 20

# Define response models
class SearchResult(BaseModel):
    search_term: str
    match_type: str
    avg_cpm: Any
    impressions: Any
    interactions: Any
    conversions: Any

class PredictionResponse(BaseModel):
    retrieved_documents: List[SearchResult]
    prediction: Dict[str, Any]
    prediction_text: str
    query_time_seconds: float

# Global variables to store databases and dataframes
evergreen_db = None
non_evergreen_db = None
evergreen_train_df = None
non_evergreen_train_df = None

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
        document = Document(
            page_content=content,
            metadata={
                "row": idx,
                "category": category,
                "match_type": match_type
            }
        )
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
                import shutil
                shutil.rmtree(db_name, ignore_errors=True)
    
    # If we get here, we need to build the database
    if not documents:
        raise ValueError("Documents must be provided to create a new database")
    
    # Remove existing database if it exists
    if os.path.exists(db_name):
        import shutil
        shutil.rmtree(db_name, ignore_errors=True)
        print(f"Removed existing database at {db_name}")
    
    # Create new database with progress indicators
    print(f"Creating new vector database with {len(documents)} documents...")
    
    # Process in batches to show progress
    batch_size = max(1, len(documents) // 10)  # Show progress for 10 batches
    all_batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
    
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
    return results, end_time - start_time

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
        reasoning = []
        
        in_reasoning = False
        
        for line in lines:
            if line.startswith("Reasoning:"):
                in_reasoning = True
                continue
                
            if in_reasoning and line.strip().startswith("•"):
                reasoning.append(line.strip())
                continue
                
            if "Predicted Average CPM:" in line:
                value = line.split(":", 1)[1].strip()
                try:
                    metrics["avg_cpm"] = float(value)
                except ValueError:
                    metrics["avg_cpm"] = value
            elif "Predicted Impressions:" in line:
                value = line.split(":", 1)[1].strip()
                try:
                    metrics["impressions"] = float(value)
                except ValueError:
                    metrics["impressions"] = value
            elif "Predicted Interactions:" in line:
                value = line.split(":", 1)[1].strip()
                try:
                    metrics["interactions"] = float(value)
                except ValueError:
                    metrics["interactions"] = value
            elif "Predicted Conversions:" in line:
                value = line.split(":", 1)[1].strip()
                try:
                    metrics["conversions"] = float(value)
                except ValueError:
                    metrics["conversions"] = value
        
        metrics["reasoning"] = reasoning
        
        return metrics
    except Exception as e:
        print(f"Error parsing prediction: {e}")
        return {}

# Initialize databases and dataframes on startup
@app.on_event("startup")
async def startup_event():
    global evergreen_db, non_evergreen_db, evergreen_train_df, non_evergreen_train_df
    
    # Load and split data
    evergreen_train_df, non_evergreen_train_df, _, _ = load_and_split_data("Search_terms.csv", test_size=0.001)
    
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

# API endpoint for search and prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: SearchRequest):
    global evergreen_db, non_evergreen_db, evergreen_train_df, non_evergreen_train_df
    
    # Check if databases are loaded
    if evergreen_db is None or non_evergreen_db is None:
        raise HTTPException(status_code=500, detail="Databases not initialized. Please try again later.")
    
    # Query the appropriate database
    results, query_time = query_database(
        evergreen_db, 
        non_evergreen_db, 
        request.query, 
        request.is_evergreen, 
        match_type=request.match_type, 
        k=request.k
    )
    
    # Extract metrics from results
    formatted_results = extract_metrics_from_results(results, evergreen_train_df, non_evergreen_train_df)
    
    # Predict metrics
    prediction_text = predict_metrics_with_llm(request.query, formatted_results, request.match_type)
    
    # Parse prediction
    parsed_prediction = parse_prediction(prediction_text)
    
    # Return response
    return PredictionResponse(
        retrieved_documents=formatted_results,
        prediction=parsed_prediction,
        prediction_text=prediction_text,
        query_time_seconds=query_time
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "databases_loaded": evergreen_db is not None and non_evergreen_db is not None}

# Root endpoint with API info
@app.get("/")
async def root():
    return {
        "api": "RAG Prediction API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Get predictions for a search term",
            "/health": "GET - Check API health"
        }
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
    
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))  # fallback to 10000 for local dev
    uvicorn.run(app, host="0.0.0.0", port=port)

