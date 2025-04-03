import pandas as pd
import os
import shutil
import time
import openai
from dotenv import load_dotenv
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

# Function to load documents
def load_documents(file_path):
    print(f"Loading documents from {file_path}...")
    if file_path.endswith('.csv'):
        # Use CSVLoader which treats each row as a separate document
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} rows from CSV as individual documents")
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path=file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} text documents")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return documents

# Function to create or load vector database
def create_vector_db(documents=None, db_name="faiss_index", force_rebuild=False):
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
    if documents is None:
        raise ValueError("Documents must be provided to create a new database")
    
    # Remove existing database if it exists
    if os.path.exists(db_name):
        shutil.rmtree(db_name, ignore_errors=True)
        print(f"Removed existing database at {db_name}")
    
    # Create new database with progress indicators
    print(f"Creating new vector database with {len(documents)} documents...")
    print("This may take a while for large datasets...")
    
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

# Function to query the database
def query_database(db, query, k=20):
    print(f"Querying database with: '{query}'")
    start_time = time.time()
    results = db.similarity_search(query, k=k)
    end_time = time.time()
    print(f"Query completed in {end_time - start_time:.2f} seconds")
    return results

# Function to format and display search results in a readable way
def display_search_results(results):
    print("\n===== SEARCH RESULTS =====")
    
    # Load the CSV file once for reference
    try:
        df = pd.read_csv("Search_terms.csv")
        csv_loaded = True
    except Exception as e:
        print(f"Warning: Could not load CSV file: {e}")
        csv_loaded = False
    
    # Create a list to store formatted results for LLM
    formatted_results = []
    
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        
        # Extract content
        content = doc.page_content
        
        # Initialize variables for key metrics
        search_term = "N/A"
        match_type = "N/A"
        avg_cpm = "N/A"
        impressions = "N/A"
        interactions = "N/A"
        conversions = "N/A"
        
        # Try to get the original row data from metadata
        if csv_loaded and 'row' in doc.metadata:
            row_num = doc.metadata['row']
            
            # Get the full row data
            if 0 <= row_num < len(df):
                row_data = df.iloc[row_num]
                
                # Extract the search term from the first column
                if 'Search terms report' in row_data.index or 'Search term' in row_data.index:
                    search_term = row_data.get('Search terms report', row_data.get('Search term', 'N/A'))
                elif len(row_data) > 0:
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
                
                # Print the search term and key metrics only
                print(f"Search Term: {search_term}")
        else:
            # Try to extract search term from content if metadata approach failed
            if "Search terms report:" in content:
                parts = content.split("Search terms report:", 1)
                if len(parts) > 1:
                    search_term_part = parts[1].strip()
                    if "\n" in search_term_part:
                        search_term = search_term_part.split("\n")[0].strip()
                    else:
                        search_term = search_term_part
        
        # Highlight key metrics
        print("\n🔍 KEY METRICS SUMMARY:")
        print(f"  Search Term: {search_term}")
        print(f"  🎯 Match Type: {match_type}")
        print(f"  💰 Avg. CPM: {avg_cpm}")
        print(f"  👁️ Impressions: {impressions}")
        print(f"  🖱️ Interactions: {interactions}")
        print(f"  🎯 Conversions: {conversions}")
        
        # Add to formatted results for LLM
        formatted_results.append({
            "search_term": search_term,
            "match_type": match_type,
            "avg_cpm": avg_cpm,
            "impressions": impressions,
            "interactions": interactions,
            "conversions": conversions
        })
    
    print("\n=========================")
    return formatted_results

# Function to predict metrics using OpenAI
def predict_metrics_with_llm(query, similar_results):
    print("\n🧠 Analyzing data with AI to predict metrics for your search term...")
    
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

# Interactive search function with LLM prediction
def interactive_search_with_prediction(db):
    while True:
        query = input("\nEnter a search term (or 'quit' to exit): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        # Perform search to get similar terms
        results = query_database(db, query, k=20)
        
        # Display results and get formatted data
        formatted_results = display_search_results(results)
        
        # Use LLM to predict metrics
        prediction = predict_metrics_with_llm(query, formatted_results)
        
        # Display the prediction
        print("\n🔮 PREDICTED METRICS FOR YOUR SEARCH TERM:")
        print("=" * 50)
        print(prediction)
        print("=" * 50)

# Main execution
if __name__ == "__main__":
    # Set this to True to force rebuilding the database, False to use existing if available
    FORCE_REBUILD = False
    
    db_name = "faiss_index"
    file_path = "Search_terms.csv"
    
    # Try to load existing database first if not forcing rebuild
    if not FORCE_REBUILD and os.path.exists(db_name):
        try:
            db = create_vector_db(db_name=db_name, force_rebuild=False)
        except Exception as e:
            print(f"Could not load existing database: {e}")
            print("Loading documents and creating new database...")
            documents = load_documents(file_path)
            db = create_vector_db(documents, db_name=db_name, force_rebuild=True)
    else:
        # Load documents and create database
        documents = load_documents(file_path)
        db = create_vector_db(documents, db_name=db_name, force_rebuild=True)
    
    # Print the first 5 documents in the database as a sample
    print("\nSample of first 5 documents in the database:")
    sample_results = db.similarity_search("", k=5)
    display_search_results(sample_results)
    
    # Start interactive search with prediction
    print("\n\n=== Starting Interactive Search with AI Prediction ===")
    print("Now you can search for terms and get predicted metrics based on similar terms")
    interactive_search_with_prediction(db)
