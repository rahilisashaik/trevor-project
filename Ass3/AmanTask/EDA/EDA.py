import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from io import StringIO
from contextlib import redirect_stdout

# Create a function to capture and write output to both console and file
class OutputCapture:
    def __init__(self, output_file):
        self.output_file = output_file
        self.buffer = StringIO()
        self.original_stdout = sys.stdout
        
    def __enter__(self):
        sys.stdout = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        
    def write(self, text):
        self.original_stdout.write(text)
        self.buffer.write(text)
        
    def flush(self):
        self.original_stdout.flush()
        
    def save_to_file(self):
        with open(self.output_file, 'w') as f:
            f.write(self.buffer.getvalue())
        print(f"\nOutput saved to {self.output_file}")

# Load the data
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
        
        # Print first few rows to understand the structure
        print("\nFirst few rows of the data:")
        print(df.head())
        
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
            else:
                # Try to rename columns based on common patterns
                print("Could not find header row, attempting to clean columns manually")
                # This would need customization based on the actual file structure
        
        # Print the actual column names after cleaning
        print("\nActual column names:")
        print(df.columns.tolist())
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Clean the data by removing rows with low impressions
def clean_data(df, min_impressions=5):
    """Remove rows with fewer impressions than the threshold."""
    if df is None:
        return None
    
    # Count rows before filtering
    rows_before = df.shape[0]
    
    # Look for the impressions column - specifically 'Impr.'
    impressions_col = 'Impr.'
    if impressions_col not in df.columns:
        # Try to find any column that might contain impression information
        for col in df.columns:
            if 'impr' in str(col).lower():
                impressions_col = col
                break
        
        if impressions_col not in df.columns:
            print("Could not find impressions column. Available columns:")
            print(df.columns.tolist())
            return df
    
    print(f"Using column '{impressions_col}' for impressions filtering")
    
    # Convert impressions to numeric, coercing errors to NaN
    df[impressions_col] = pd.to_numeric(df[impressions_col], errors='coerce')
    
    # Filter rows with at least min_impressions
    df_cleaned = df[df[impressions_col] >= min_impressions]
    
    # Count rows after filtering
    rows_after = df_cleaned.shape[0]
    rows_removed = rows_before - rows_after
    
    print(f"Removed {rows_removed} rows ({rows_removed/rows_before:.2%}) with fewer than {min_impressions} impressions.")
    print(f"Remaining rows: {rows_after}")
    
    return df_cleaned

# Remove sparse rows (rows with many missing values)
def remove_sparse_rows(df, threshold=0.7):
    """Remove rows with more than threshold proportion of missing values."""
    if df is None:
        return None
    
    # Count rows before filtering
    rows_before = df.shape[0]
    
    # Calculate the proportion of non-missing values for each row
    completeness = df.count(axis=1) / df.shape[1]
    
    # Keep rows with completeness above threshold
    df_cleaned = df[completeness >= threshold]
    
    # Count rows after filtering
    rows_after = df_cleaned.shape[0]
    rows_removed = rows_before - rows_after
    
    print(f"Removed {rows_removed} sparse rows ({rows_removed/rows_before:.2%}) with more than {(1-threshold):.0%} missing values.")
    print(f"Remaining rows: {rows_after}")
    
    return df_cleaned

# Find top 5 search terms with lowest Avg. CPM for each Ad Group
def top_5_lowest_cpm_by_adgroup(df):
    """For each Ad Group, find the top 5 Search Terms with the lowest Avg. CPM."""
    if df is None:
        return None
    
    # Find the relevant columns - use exact names from the dataset
    ad_group_col = 'Ad group'
    search_term_col = 'Search term'
    cpm_col = 'Avg. CPM'
    
    # Check if columns exist
    missing_cols = []
    for col_name, col in zip(['Ad group', 'Search term', 'Avg. CPM'], 
                            [ad_group_col, search_term_col, cpm_col]):
        if col not in df.columns:
            missing_cols.append(col_name)
    
    if missing_cols:
        print(f"Missing required columns: {', '.join(missing_cols)}")
        print("Available columns:")
        print(df.columns.tolist())
        
        # Try to find alternative column names
        for col_name in missing_cols:
            for df_col in df.columns:
                if col_name.lower() in str(df_col).lower():
                    if col_name == 'Ad group':
                        ad_group_col = df_col
                    elif col_name == 'Search term':
                        search_term_col = df_col
                    elif col_name == 'Avg. CPM':
                        cpm_col = df_col
                    print(f"Using '{df_col}' for '{col_name}'")
                    break
    
    if not all([ad_group_col in df.columns, search_term_col in df.columns, cpm_col in df.columns]):
        print("Could not find all required columns")
        return None
    
    print(f"Using columns: '{ad_group_col}', '{search_term_col}', '{cpm_col}'")
    
    # Convert CPM to numeric, coercing errors to NaN
    # Remove $ and , from the values if they're strings
    if df[cpm_col].dtype == object:
        df[cpm_col] = df[cpm_col].astype(str).str.replace('$', '').str.replace(',', '')
    df[cpm_col] = pd.to_numeric(df[cpm_col], errors='coerce')
    
    # Group by Ad Group and find the 5 search terms with lowest CPM
    result = {}
    for ad_group, group_df in df.groupby(ad_group_col):
        # Skip groups with no valid CPM values
        if group_df[cpm_col].isna().all():
            print(f"Skipping Ad Group '{ad_group}' - no valid CPM values")
            continue
            
        # Sort by CPM (ascending) and take top 5
        top_5 = group_df.sort_values(cpm_col).head(5)
        result[ad_group] = top_5
        
        # Print the results
        print(f"\nTop 5 Search Terms with lowest Avg. CPM for Ad Group '{ad_group}':")
        print(top_5[[search_term_col, cpm_col]].to_string(index=False))
    
    return result

# Find Ad Group with the largest number of rows
def largest_adgroup(df):
    """Find the Ad Group with the largest number of rows."""
    if df is None:
        return None
    
    # Find the Ad Group column
    ad_group_col = 'Ad group'
    if ad_group_col not in df.columns:
        # Try to find any column that might contain ad group information
        for col in df.columns:
            if 'ad group' in str(col).lower():
                ad_group_col = col
                break
        
        if ad_group_col not in df.columns:
            print("Could not find Ad Group column. Available columns:")
            print(df.columns.tolist())
            return None
    
    # Count rows by Ad Group
    adgroup_counts = df[ad_group_col].value_counts()
    
    if adgroup_counts.empty:
        print("No Ad Groups found in the data")
        return None
    
    # Get the Ad Group with the most rows
    largest_group = adgroup_counts.index[0]
    count = adgroup_counts.iloc[0]
    
    print(f"\nAd Group with the largest number of rows: '{largest_group}' with {count} rows")
    
    return largest_group, count

# Find dataset with most non-sparse rows
def find_most_complete_dataset(df):
    """Find the Ad Group with the most non-sparse rows (fewest missing values)."""
    if df is None:
        return None
    
    # Find the Ad Group column
    ad_group_col = None
    for col in df.columns:
        if 'ad group' in str(col).lower():
            ad_group_col = col
            break
    
    if ad_group_col is None:
        print("Could not find Ad Group column")
        return None
    
    # Group by Ad Group and calculate missing values
    result = {}
    for ad_group, group_df in df.groupby(ad_group_col):
        # Calculate percentage of non-missing values for each column
        non_missing_pct = group_df.count() / len(group_df)
        # Calculate average completeness across all columns
        avg_completeness = non_missing_pct.mean()
        # Store result
        result[ad_group] = {
            'row_count': len(group_df),
            'avg_completeness': avg_completeness,
            'non_missing_values': group_df.count().sum(),
            'total_possible_values': group_df.size
        }
    
    # Find the Ad Group with highest completeness
    if not result:
        print("No Ad Groups found with valid data")
        return None
    
    # Sort by completeness and row count
    sorted_groups = sorted(result.items(), 
                          key=lambda x: (x[1]['avg_completeness'], x[1]['row_count']), 
                          reverse=True)
    
    best_group = sorted_groups[0][0]
    stats = result[best_group]
    
    print(f"\nAd Group with most complete data: '{best_group}'")
    print(f"  Row count: {stats['row_count']}")
    print(f"  Average completeness: {stats['avg_completeness']:.2%}")
    print(f"  Non-missing values: {stats['non_missing_values']} out of {stats['total_possible_values']} ({stats['non_missing_values']/stats['total_possible_values']:.2%})")
    
    return best_group, stats

# Visualize the distribution of impressions
def visualize_impressions(df, original_df=None):
    """Create visualizations for the impressions distribution."""
    if df is None:
        return
    
    # Find the impressions column - specifically 'Impr.'
    impressions_col = 'Impr.'
    if impressions_col not in df.columns:
        # Try to find any column that might contain impression information
        for col in df.columns:
            if 'impr' in str(col).lower():
                impressions_col = col
                break
        
        if impressions_col not in df.columns:
            print("Could not find impressions column for visualization")
            return
    
    # Ensure impressions are numeric
    df[impressions_col] = pd.to_numeric(df[impressions_col], errors='coerce')
    
    plt.figure(figsize=(12, 6))
    
    # If we have both original and cleaned dataframes, show before/after
    if original_df is not None:
        original_df[impressions_col] = pd.to_numeric(original_df[impressions_col], errors='coerce')
        
        plt.subplot(1, 2, 1)
        sns.histplot(original_df[impressions_col].dropna(), bins=50, kde=True)
        plt.title('Impressions Distribution (Before Cleaning)')
        plt.xlabel('Impressions')
        plt.ylabel('Count')
        plt.xscale('log')
        
        plt.subplot(1, 2, 2)
        sns.histplot(df[impressions_col].dropna(), bins=50, kde=True)
        plt.title('Impressions Distribution (After Cleaning)')
        plt.xlabel('Impressions')
        plt.ylabel('Count')
        plt.xscale('log')
    else:
        # Just show the cleaned data
        sns.histplot(df[impressions_col].dropna(), bins=50, kde=True)
        plt.title('Impressions Distribution')
        plt.xlabel('Impressions')
        plt.ylabel('Count')
        plt.xscale('log')
    
    plt.tight_layout()
    
    # Save the plot in the current directory
    plt.savefig('impressions_distribution.png')
    print("\nSaved impressions distribution plot to 'impressions_distribution.png'")

# Main function to run the EDA
def main():
    # Set up output capture - save in current directory
    with OutputCapture('Results/eda_results.txt') as output:
        # Load the data
        df = load_data()
        
        if df is not None:
            # Keep a copy of the original data for comparison
            df_original = df.copy()
            
            # Display basic info about the data
            print("\nBasic information about the dataset:")
            print(f"Shape: {df.shape}")
            print("\nSummary statistics:")
            print(df.describe())
            
            # Clean the data - remove rows with fewer than 5 impressions
            df_cleaned = clean_data(df, min_impressions=5)
            
            if df_cleaned is not None and not df_cleaned.empty:
                # Remove sparse rows
                df_cleaned = remove_sparse_rows(df_cleaned, threshold=0.7)  # Keep rows with at least 70% non-missing values
                
                if df_cleaned is not None and not df_cleaned.empty:
                    # Find top 5 search terms with lowest CPM for each Ad Group
                    top_5_results = top_5_lowest_cpm_by_adgroup(df_cleaned)
                    
                    # Find Ad Group with the largest number of rows after cleaning
                    largest_group_result = largest_adgroup(df_cleaned)
                    
                    # Find dataset with most non-sparse rows
                    most_complete_dataset = find_most_complete_dataset(df_cleaned)
                    
                    # Visualize the impressions distribution
                    visualize_impressions(df_cleaned, df_original)
                    
                    print("\nEDA completed successfully!")
                else:
                    print("No data left after removing sparse rows. Consider using a lower threshold.")
            else:
                print("No data left after cleaning. Consider using a lower threshold for minimum impressions.")
        
        # Save the captured output to file
        output.save_to_file()

if __name__ == "__main__":
    main()
