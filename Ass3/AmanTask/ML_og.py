import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor  # Replacing Naive Bayes
from sklearn.svm import SVR
import xgboost as xgb
from gensim.models import Word2Vec
import nltk
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("Please run 'python -m nltk.downloader punkt' in your terminal")

# Set random seed for reproducibility
np.random.seed(42)

def load_data(file_path='Search_terms.csv'):
    """Load the search terms data from CSV file."""
    try:
        # Try to find the file in the current directory
        if not os.path.exists(file_path):
            print(f"File not found at {file_path}, checking current directory")
            # List files in current directory to help debug
            print(f"Files in current directory: {os.listdir('.')}")
            
        # Read with low_memory=False to avoid DtypeWarning
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # Clean up column names and skip header rows if needed
        if 'Search terms report' in df.columns:
            # Check if the first rows contain actual column headers
            potential_headers = df.iloc[0:5].values.tolist()
            header_row = None
            
            # Look for rows that might contain actual column headers
            for i, row in enumerate(potential_headers):
                if any(col in str(val).lower() for val in row for col in ['impressions', 'cpm', 'ad group']):
                    header_row = i
                    break
            
            if header_row is not None:
                # Use the identified row as header
                new_headers = df.iloc[header_row].values
                df = df.iloc[header_row+1:].reset_index(drop=True)
                df.columns = new_headers
                print(f"Used row {header_row} as column headers")
        
        # Print the actual column names after cleaning
        print("\nActual column names:")
        print(df.columns.tolist())
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df, min_impressions=5):
    """Clean the data by removing rows with low impressions and sparse rows."""
    if df is None:
        return None
    
    # Find the impressions column - specifically 'Impr.'
    impressions_col = 'Impr.'
    if impressions_col not in df.columns:
        # Try to find any column that might contain impression information
        for col in df.columns:
            if 'impr' in str(col).lower():
                impressions_col = col
                break
    
    # Convert impressions to numeric, coercing errors to NaN
    df[impressions_col] = pd.to_numeric(df[impressions_col], errors='coerce')
    
    # Filter rows with at least min_impressions
    df_cleaned = df[df[impressions_col] >= min_impressions]
    
    # Remove sparse rows (rows with many missing values)
    # Calculate the proportion of non-missing values for each row
    completeness = df_cleaned.count(axis=1) / df_cleaned.shape[1]
    
    # Keep rows with completeness above threshold (70%)
    df_cleaned = df_cleaned[completeness >= 0.7]
    
    return df_cleaned

def get_ad_group_data(df, ad_group_name='Supportive Adults Prospecting'):
    """Extract data for the specified Ad Group."""
    # Find the Ad Group column
    ad_group_col = 'Ad group'
    if ad_group_col not in df.columns:
        # Try to find any column that might contain ad group information
        for col in df.columns:
            if 'ad group' in str(col).lower():
                ad_group_col = col
                break
    
    # Filter data for the specified Ad Group
    ad_group_data = df[df[ad_group_col] == ad_group_name]
    print(f"Extracted {len(ad_group_data)} rows for Ad Group '{ad_group_name}'")
    
    return ad_group_data

def preprocess_text(text):
    """Clean and tokenize text."""
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Simple tokenization (split by whitespace) to avoid NLTK dependency issues
    tokens = text.split()
    
    # Remove short tokens
    tokens = [token for token in tokens if len(token) > 2]
    
    return tokens

def train_word2vec(texts, vector_size=100, window=5, min_count=1, workers=4):
    """Train a Word2Vec model on the given texts."""
    # Tokenize texts
    tokenized_texts = [preprocess_text(text) for text in texts]
    
    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, 
                     window=window, min_count=min_count, workers=workers)
    
    print(f"Word2Vec model trained with vocabulary size: {len(model.wv.key_to_index)}")
    
    return model, tokenized_texts

def get_text_embedding(text, word2vec_model, vector_size=100):
    """Convert text to embedding using Word2Vec model."""
    if not isinstance(text, str):
        return np.zeros(vector_size)
    
    tokens = preprocess_text(text)
    
    if not tokens:
        return np.zeros(vector_size)
    
    # Get embeddings for each token in the text
    token_embeddings = []
    for token in tokens:
        if token in word2vec_model.wv:
            token_embeddings.append(word2vec_model.wv[token])
    
    # If no tokens have embeddings, return zeros
    if not token_embeddings:
        return np.zeros(vector_size)
    
    # Average the embeddings
    return np.mean(token_embeddings, axis=0)

def extract_text_features(text):
    """Extract features from text."""
    if not isinstance(text, str):
        return {
            'word_count': 0,
            'char_count': 0,
            'has_question': 0,
            'has_exclamation': 0,
            'has_numbers': 0
        }
    
    features = {
        'word_count': len(text.split()),
        'char_count': len(text),
        'has_question': 1 if '?' in text else 0,
        'has_exclamation': 1 if '!' in text else 0,
        'has_numbers': 1 if any(char.isdigit() for char in text) else 0
    }
    
    return features

def engineer_features(df, word2vec_model, search_term_col='Search term', vector_size=100):
    """Engineer features for ML models."""
    # Create a copy of the dataframe
    df_features = df.copy()
    
    # Convert target columns to numeric
    target_cols = ['Avg. CPM', 'Impr.', 'Interactions']
    for col in target_cols:
        if col in df_features.columns:
            if df_features[col].dtype == object:
                # Remove $ and , from the values if they're strings
                df_features[col] = df_features[col].astype(str).str.replace('$', '').str.replace(',', '')
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
    
    # Drop rows with missing target values
    df_features = df_features.dropna(subset=[col for col in target_cols if col in df_features.columns])
    
    # Get Word2Vec embeddings for search terms
    print("Generating Word2Vec embeddings for search terms...")
    embeddings = []
    for text in df_features[search_term_col]:
        embedding = get_text_embedding(text, word2vec_model, vector_size)
        embeddings.append(embedding)
    
    # Convert embeddings to DataFrame
    embedding_cols = [f'embedding_{i}' for i in range(vector_size)]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)
    
    # Extract text features
    print("Extracting text features...")
    text_features = []
    for text in df_features[search_term_col]:
        features = extract_text_features(text)
        text_features.append(features)
    
    # Convert text features to DataFrame
    text_features_df = pd.DataFrame(text_features)
    
    # Combine all features
    df_features = pd.concat([df_features.reset_index(drop=True), 
                            embeddings_df.reset_index(drop=True),
                            text_features_df.reset_index(drop=True)], axis=1)
    
    print(f"Engineered features dataframe shape: {df_features.shape}")
    
    return df_features

def prepare_train_test_data(df, target_col, test_size=0.2):
    """Prepare training and testing data for a specific target."""
    # Identify feature columns (all embedding and text feature columns)
    feature_cols = [col for col in df.columns if col.startswith('embedding_') or 
                   col in ['word_count', 'char_count', 'has_question', 'has_exclamation', 'has_numbers']]
    
    # Split data into features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Prepared data for target '{target_col}':")
    print(f"  Training set: {X_train_scaled.shape[0]} samples")
    print(f"  Testing set: {X_test_scaled.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name, target_col):
    """Evaluate model performance."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate percentage deviation
    # Avoid division by zero
    mask = y_test != 0
    if mask.sum() > 0:
        percentage_deviation = 100 * np.abs(y_pred[mask] - y_test[mask]) / y_test[mask]
        mean_percentage_deviation = np.mean(percentage_deviation)
    else:
        mean_percentage_deviation = np.nan
    
    print(f"{model_name} - {target_col}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Mean Percentage Deviation: {mean_percentage_deviation:.2f}%")
    
    return {
        'model_name': model_name,
        'target_col': target_col,
        'rmse': rmse,
        'mean_percentage_deviation': mean_percentage_deviation
    }

def train_and_evaluate_models(df_features, target_cols):
    """Train and evaluate multiple ML models for each target."""
    results = []
    
    for target_col in target_cols:
        print(f"\n=== Training models for target: {target_col} ===")
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_train_test_data(df_features, target_col)
        
        # 1. Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_result = evaluate_model(rf_model, X_test, y_test, "Random Forest", target_col)
        results.append(rf_result)
        
        # 2. XGBoost
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_result = evaluate_model(xgb_model, X_test, y_test, "XGBoost", target_col)
        results.append(xgb_result)
        
        # 3. KNN
        print("\nTraining KNN...")
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        knn_result = evaluate_model(knn_model, X_test, y_test, "KNN", target_col)
        results.append(knn_result)
        
        # 4. Decision Tree (replacing Naive Bayes)
        print("\nTraining Decision Tree (replacing Naive Bayes)...")
        try:
            dt_model = DecisionTreeRegressor(random_state=42)
            dt_model.fit(X_train, y_train)
            dt_result = evaluate_model(dt_model, X_test, y_test, "Decision Tree", target_col)
            results.append(dt_result)
        except Exception as e:
            print(f"Error training Decision Tree: {e}")
            # Add a placeholder result
            results.append({
                'model_name': "Decision Tree",
                'target_col': target_col,
                'rmse': float('inf'),
                'mean_percentage_deviation': float('inf')
            })
        
        # 5. Support Vector Regression (SVR)
        print("\nTraining SVR...")
        try:
            svr_model = SVR(kernel='rbf')
            svr_model.fit(X_train, y_train)
            svr_result = evaluate_model(svr_model, X_test, y_test, "SVR", target_col)
            results.append(svr_result)
        except Exception as e:
            print(f"Error training SVR: {e}")
            # Add a placeholder result
            results.append({
                'model_name': "SVR",
                'target_col': target_col,
                'rmse': float('inf'),
                'mean_percentage_deviation': float('inf')
            })
    
    return results

def plot_results(results):
    """Plot model performance results."""
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Remove infinite values
    results_df = results_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if results_df.empty:
        print("No valid results to plot")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot RMSE
    sns.barplot(x='model_name', y='rmse', hue='target_col', data=results_df, ax=ax1)
    ax1.set_title('RMSE by Model and Target')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot Mean Percentage Deviation
    sns.barplot(x='model_name', y='mean_percentage_deviation', hue='target_col', data=results_df, ax=ax2)
    ax2.set_title('Mean Percentage Deviation by Model and Target')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Mean Percentage Deviation (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('OG_Results/model_performance.png')
    print("\nSaved model performance plot to 'OG_Results/model_performance.png'")
    
    return results_df

def save_results_to_file(results, filename='OG_Results/model_results.csv'):
    """Save model results to a CSV file."""
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Replace infinite values with NaN
    results_df = results_df.replace([np.inf, -np.inf], np.nan)
    
    # Save to CSV
    results_df.to_csv(filename, index=False)
    print(f"\nSaved model results to '{filename}'")
    
    # Create a more readable table format
    # Pivot the data to create a table with models as rows and metrics as columns
    table_data = []
    for target in results_df['target_col'].unique():
        target_results = results_df[results_df['target_col'] == target]
        for _, row in target_results.iterrows():
            table_data.append({
                'Target': row['target_col'],
                'Model': row['model_name'],
                'RMSE': f"{row['rmse']:.4f}",
                'Mean % Deviation': f"{row['mean_percentage_deviation']:.2f}%"
            })
    
    # Convert to DataFrame and save as a formatted table
    table_df = pd.DataFrame(table_data)
    
    # Save as a more readable table
    with open('OG_Results/model_results_table.txt', 'w') as f:
        f.write("Model Performance Results\n")
        f.write("=======================\n\n")
        
        # Group by target
        for target in table_df['Target'].unique():
            f.write(f"Target: {target}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Model':<20} {'RMSE':<15} {'Mean % Deviation':<20}\n")
            f.write("-" * 60 + "\n")
            
            # Get results for this target
            target_table = table_df[table_df['Target'] == target]
            for _, row in target_table.iterrows():
                f.write(f"{row['Model']:<20} {row['RMSE']:<15} {row['Mean % Deviation']:<20}\n")
            
            f.write("\n\n")
    
    print(f"Saved formatted results table to 'OG_Results/model_results_table.txt'")
    
    return results_df

def main():
    """Main function to run the ML benchmarking."""
    print("=== ML Model Benchmarking ===")
    
    # Load data
    df = load_data()
    
    if df is None:
        print("Error: Could not load data.")
        return
    
    # Clean data
    df_cleaned = clean_data(df)
    
    if df_cleaned is None or df_cleaned.empty:
        print("Error: No data left after cleaning.")
        return
    
    # Get data for the Ad Group with the largest number of non-sparse rows
    ad_group_data = get_ad_group_data(df_cleaned, 'Supportive Adults Prospecting')
    
    if ad_group_data.empty:
        print("Error: No data found for the specified Ad Group.")
        return
    
    # Train Word2Vec model on search terms
    search_term_col = 'Search term'
    if search_term_col not in ad_group_data.columns:
        # Try to find the search term column
        for col in ad_group_data.columns:
            if 'search term' in str(col).lower():
                search_term_col = col
                break
    
    print(f"Using column '{search_term_col}' for search terms")
    
    word2vec_model, tokenized_texts = train_word2vec(ad_group_data[search_term_col])
    
    # Engineer features
    df_features = engineer_features(ad_group_data, word2vec_model, search_term_col)
    
    # Identify target columns
    target_cols = []
    for col_name in ['Avg. CPM', 'Impr.', 'Interactions']:
        if col_name in df_features.columns:
            target_cols.append(col_name)
        else:
            # Try to find alternative column names
            for col in df_features.columns:
                if col_name.lower() in str(col).lower():
                    target_cols.append(col)
                    print(f"Using '{col}' for '{col_name}'")
                    break
    
    print(f"Target columns: {target_cols}")
    
    # Train and evaluate models
    results = train_and_evaluate_models(df_features, target_cols)
    
    # Plot results
    results_df = plot_results(results)
    
    # Save results to file
    save_results_to_file(results)
    
    # Find best model for each target
    best_models = {}
    for target_col in target_cols:
        target_results = [r for r in results if r['target_col'] == target_col]
        if target_results:
            best_model = min(target_results, key=lambda x: x['rmse'])
            best_models[target_col] = best_model
    
    print("\n=== Best Models ===")
    for target_col, model in best_models.items():
        print(f"Best model for {target_col}:")
        print(f"  Model: {model['model_name']}")
        print(f"  RMSE: {model['rmse']:.4f}")
        print(f"  Mean Percentage Deviation: {model['mean_percentage_deviation']:.2f}%")

if __name__ == "__main__":
    main()
