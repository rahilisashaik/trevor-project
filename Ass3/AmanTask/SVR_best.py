import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from gensim.models import Word2Vec
import nltk
import re
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Set random seed for reproducibility
np.random.seed(42)

# Create Final_Results directory if it doesn't exist
os.makedirs('Final_Results', exist_ok=True)

# Define the best parameters for each target based on previous experiments
BEST_PARAMS = {
    'Avg. CPM': {
        'kernel': 'rbf',
        'C': 10.0,
        'gamma': 'scale',
        'epsilon': 0.1
    },
    'Impr.': {
        'kernel': 'rbf',
        'C': 100.0,
        'gamma': 'scale',
        'epsilon': 0.1
    },
    'Interactions': {
        'kernel': 'rbf',
        'C': 10.0,
        'gamma': 'auto',
        'epsilon': 0.2
    }
}

def load_data(file_path='Search_terms.csv'):
    """Load the search terms data from CSV file."""
    try:
        # Try to find the file in the current directory or parent directory
        if not os.path.exists(file_path):
            if os.path.exists(f"../Search_terms.csv"):
                file_path = f"../Search_terms.csv"
            else:
                print(f"File not found at {file_path}, checking current directory")
                print(f"Files in current directory: {os.listdir('.')}")
                return None
            
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
    
    # Simple tokenization (split by whitespace)
    tokens = text.split()
    
    # Remove short tokens
    tokens = [token for token in tokens if len(token) > 2]
    
    return tokens

def train_word2vec(texts, vector_size=100, window=5, min_count=1, workers=4):
    """Train a Word2Vec model on the given texts."""
    # Tokenize texts
    tokenized_texts = [preprocess_text(text) for text in texts]
    
    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_texts, 
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count, 
                     workers=workers,
                     sg=1)  # Use skip-gram
    
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
            'has_numbers': 0,
            'uppercase_ratio': 0
        }
    
    features = {
        'word_count': len(text.split()),
        'char_count': len(text),
        'has_question': 1 if '?' in text else 0,
        'has_exclamation': 1 if '!' in text else 0,
        'has_numbers': 1 if any(char.isdigit() for char in text) else 0,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1)
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
    
    # Generate Word2Vec embeddings for search terms
    print("Generating Word2Vec embeddings for search terms...")
    embeddings = []
    for text in df_features[search_term_col]:
        embedding = get_text_embedding(text, word2vec_model, vector_size)
        embeddings.append(embedding)
    
    # Convert embeddings to DataFrame
    embedding_cols = [f'embedding_{i}' for i in range(vector_size)]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    
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
                            embedding_df.reset_index(drop=True),
                            text_features_df.reset_index(drop=True)], axis=1)
    
    print(f"Engineered features dataframe shape: {df_features.shape}")
    
    return df_features

def prepare_train_test_data(df, target_col, test_size=0.2, random_state=42):
    """Prepare training and testing data for a specific target."""
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataframe")
        return None, None, None, None
    
    # Drop rows with missing target values
    df_valid = df.dropna(subset=[target_col])
    
    if df_valid.empty:
        print(f"No valid data for target '{target_col}' after dropping missing values")
        return None, None, None, None
    
    # Select features (all columns except the target columns)
    target_cols = ['Avg. CPM', 'Impr.', 'Interactions']
    feature_cols = [col for col in df_valid.columns if col not in target_cols]
    
    # Handle categorical columns if any
    categorical_cols = []
    for col in feature_cols:
        if df_valid[col].dtype == 'object' and col != 'Search term':
            categorical_cols.append(col)
    
    # Drop categorical columns for now (could one-hot encode them if needed)
    feature_cols = [col for col in feature_cols if col not in categorical_cols and col != 'Search term']
    
    # Select features and target
    X = df_valid[feature_cols]
    y = df_valid[target_col]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"Prepared data for target '{target_col}':")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name, target_col):
    """Evaluate a model and return performance metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate mean percentage deviation
    # Avoid division by zero by adding a small epsilon where y_test is zero
    epsilon = 1e-10
    percentage_deviation = 100 * np.abs(y_pred - y_test) / (np.abs(y_test) + epsilon)
    mean_percentage_deviation = np.mean(percentage_deviation)
    
    print(f"{model_name} - {target_col}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Mean Percentage Deviation: {mean_percentage_deviation:.2f}%")
    
    return {
        'model_name': model_name,
        'target_col': target_col,
        'rmse': rmse,
        'mean_percentage_deviation': mean_percentage_deviation,
        'y_test': y_test,
        'y_pred': y_pred
    }

def train_best_svr(X_train, X_test, y_train, y_test, target_col):
    """Train the best SVR model for the given target."""
    print(f"\n=== Training Best SVR for target: {target_col} ===")
    
    # Get the best parameters for this target
    best_params = BEST_PARAMS.get(target_col, {
        'kernel': 'rbf',
        'C': 10.0,
        'gamma': 'scale',
        'epsilon': 0.1
    })
    
    print(f"Using parameters: {best_params}")
    
    # Check for NaN values in the data
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        print("Warning: Input data contains NaN values. Imputing missing values...")
        
        # Impute missing values in X_train and X_test
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # Impute missing values in y_train if any
        if np.isnan(y_train).any():
            y_imputer = SimpleImputer(strategy='mean')
            y_train = y_imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        
        # Train SVR with best parameters on imputed data
        svr = SVR(**best_params)
        svr.fit(X_train_imputed, y_train)
        
        # Evaluate model on imputed test data
        result = evaluate_model(svr, X_test_imputed, y_test, "SVR (Best)", target_col)
    else:
        # Train SVR with best parameters
        svr = SVR(**best_params)
        svr.fit(X_train, y_train)
        
        # Evaluate model
        result = evaluate_model(svr, X_test, y_test, "SVR (Best)", target_col)
    
    # Add best parameters to result
    result['best_params'] = best_params
    
    # Save the model
    model_filename = f'Final_Results/svr_model_{target_col.replace(".", "")}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(svr, f)
    print(f"Saved best model to {model_filename}")
    
    return result, svr

def plot_predictions(result, target_col):
    """Plot actual vs predicted values."""
    y_test = result['y_test']
    y_pred = result['y_pred']
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of actual vs predicted
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Actual vs Predicted: {target_col}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Add RMSE and Mean Percentage Deviation to plot
    plt.annotate(f"RMSE: {result['rmse']:.4f}", xy=(0.05, 0.95), xycoords='axes fraction')
    plt.annotate(f"Mean % Deviation: {result['mean_percentage_deviation']:.2f}%", xy=(0.05, 0.90), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig(f'Final_Results/svr_predictions_{target_col.replace(".", "")}.png')
    print(f"Saved prediction plot to 'Final_Results/svr_predictions_{target_col.replace('.', '')}.png'")

def save_results_to_file(results, filename='Final_Results/svr_best_results.csv'):
    """Save model results to a CSV file."""
    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {
            'Target': result['target_col'],
            'RMSE': result['rmse'],
            'Mean_Percentage_Deviation': result['mean_percentage_deviation'],
            'Kernel': result['best_params']['kernel'],
            'C': result['best_params']['C'],
            'Gamma': result['best_params'].get('gamma', 'N/A'),
            'Epsilon': result['best_params']['epsilon']
        }
        for result in results
    ])
    
    # Save to CSV
    results_df.to_csv(filename, index=False)
    print(f"\nSaved SVR results to '{filename}'")
    
    # Create a more readable table format
    with open('Final_Results/svr_best_results_table.txt', 'w') as f:
        f.write("Best SVR Model Performance Results\n")
        f.write("================================\n\n")
        
        for result in results:
            target = result['target_col']
            f.write(f"Target: {target}\n")
            f.write("-" * 60 + "\n")
            f.write(f"RMSE: {result['rmse']:.4f}\n")
            f.write(f"Mean Percentage Deviation: {result['mean_percentage_deviation']:.2f}%\n\n")
            
            f.write("Parameters:\n")
            for param, value in result['best_params'].items():
                f.write(f"  {param}: {value}\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
    
    print(f"Saved formatted results table to 'Final_Results/svr_best_results_table.txt'")
    
    return results_df

def plot_comparison_chart(results):
    """Create bar charts comparing RMSE and Mean Percentage Deviation across targets."""
    # Extract data for plotting
    targets = [r['target_col'] for r in results]
    rmse_values = [r['rmse'] for r in results]
    mpd_values = [r['mean_percentage_deviation'] for r in results]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot RMSE
    bars1 = ax1.bar(targets, rmse_values, color='skyblue')
    ax1.set_title('RMSE by Target')
    ax1.set_xlabel('Target')
    ax1.set_ylabel('RMSE')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Plot Mean Percentage Deviation
    bars2 = ax2.bar(targets, mpd_values, color='lightgreen')
    ax2.set_title('Mean Percentage Deviation by Target')
    ax2.set_xlabel('Target')
    ax2.set_ylabel('Mean Percentage Deviation (%)')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('Final_Results/performance_comparison.png')
    print("\nSaved performance comparison chart to 'Final_Results/performance_comparison.png'")

def main():
    """Main function to run the best SVR models."""
    print("=== Best SVR Models ===")
    
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
    
    # Save the Word2Vec model
    word2vec_model.save('Final_Results/word2vec_model.model')
    print("Saved Word2Vec model to 'Final_Results/word2vec_model.model'")
    
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
    
    # Train best SVR models for each target
    results = []
    models = {}
    
    for target_col in target_cols:
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_train_test_data(df_features, target_col)
        
        if X_train is not None:
            # Train best SVR
            result, model = train_best_svr(X_train, X_test, y_train, y_test, target_col)
            results.append(result)
            models[target_col] = model
            
            # Plot predictions
            plot_predictions(result, target_col)
    
    # Save results to file
    if results:
        save_results_to_file(results)
        
        # Plot comparison chart
        plot_comparison_chart(results)
        
        print("\n=== Best SVR Models Summary ===")
        for result in results:
            print(f"Target: {result['target_col']}")
            print(f"  RMSE: {result['rmse']:.4f}")
            print(f"  Mean Percentage Deviation: {result['mean_percentage_deviation']:.2f}%")
            print(f"  Parameters: {result['best_params']}")
            print()
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
