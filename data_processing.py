import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Load the dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col=0)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the file path and try again.")
    
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty. Please provide a valid file with data.")
    
    except pd.errors.ParserError:
        print(f"Error: The file '{file_path}' could not be parsed. Please check the file format and ensure it is a valid CSV file.")
    
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")

    return None

def prepare_data(df):
    # Create a copy of the original DataFrame to work with
    df_copy = df.copy()
    
    # Remove duplicates
    df_copy.drop_duplicates(inplace=True)

    # Replace infinite values with NaN
    df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaN values
    df_copy.dropna(inplace=True)

    # Fill in missing values
    for column in df_copy.columns:
        if df_copy[column].dtype == 'object':
            # Fill missing values in categorical columns with mode
            df_copy[column].fillna(df_copy[column].mode()[0], inplace=True)
        else:
            # Fill missing values in numeric columns with mean
            df_copy[column].fillna(df_copy[column].mean(), inplace=True)

    # Convert any column with 'Date' in its name to datetime format
    for column in df_copy.columns:
        if 'date' in column.lower():
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
            # Remove the time part, retaining only the date
            df_copy[column] = df_copy[column].dt.date

    print(f"Data cleaning completed: {df.shape[0] - df_copy.shape[0]} duplicates removed, missing values filled, and date columns converted.")
    
    return df_copy

def calculate_summary_statistics(df):
    summary = df.describe()
    print("Summary Statistics:\n")
    return summary

def filter_data(df, column, value):
    filtered_df = df[df[column] == value]
    print(f"Filtered Data by {column} = {value}:\n")
    return filtered_df

def plot_histogram(df, column):
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], kde=True, bins=10)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_bar_chart(df, category_col, value_col):
    plt.figure(figsize=(10, 4))
    sns.barplot(x=category_col, y=value_col, data=df, estimator=sum, errorbar=None, palette='viridis')
    plt.title(f'Total {value_col} by {category_col}')
    plt.xlabel(category_col)
    plt.ylabel(f'Total {value_col}')
    plt.xticks(rotation=90)  # Rotate the x labels for better readability
    plt.show()

def plot_pie_chart(df, category_col, value_col, explode_index=0):
    aggregated_data = df.groupby(category_col)[value_col].sum()
    
    # Create an explode array where one slice is popped out
    explode = [0] * len(aggregated_data)
    explode[explode_index] = 0.1
    
    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(aggregated_data, labels=aggregated_data.index, autopct='%1.1f%%', 
            colors=sns.color_palette('pastel'), explode=explode)
    plt.title(f'{value_col} Distribution by {category_col}')
    plt.show()

def plot_scatter(df, x_col, y_col, hue_col=None):
    plt.figure(figsize=(10, 4))
    if hue_col:
        sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df)
    else:
        sns.scatterplot(x=x_col, y=y_col, data=df)
    plt.title(f'{y_col} vs. {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if hue_col:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()

def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process and analyze sales data.")
    parser.add_argument('file_path', type=str, help="Path to the input CSV file")
    parser.add_argument('output_path', type=str, help="Path to save the processed CSV file")
    args = parser.parse_args()
    
    #Add arguments for interactive functions
    parser.add_argument('--filter_column', type=str, help="Column to filter on")
    parser.add_argument('--filter_value', type=str, help="Value to filter by")
    
    parser.add_argument('--histogram_col', type=str, help="Column for histogram")
    parser.add_argument('--bar_category_col', type=str, help="Category column for bar chart")
    parser.add_argument('--bar_value_col', type=str, help="Value column for bar chart")
    
    parser.add_argument('--pie_category_col', type=str, help="Category column for pie chart")
    parser.add_argument('--pie_value_col', type=str, help="Value column for pie chart")
    parser.add_argument('--pie_explode_index', type=int, default=0, help="Index of pie slice to explode")
    
    parser.add_argument('--scatter_x_col', type=str, help="X-axis column for scatter plot")
    parser.add_argument('--scatter_y_col', type=str, help="Y-axis column for scatter plot")
    parser.add_argument('--scatter_hue_col', type=str, help="Hue column for scatter plot")
    
    args = parser.parse_args()
    
    original_df = load_data(args.file_path)
    if original_df is None:
        return

    # Prepare data for analysis
    cleaned_df = prepare_data(original_df)    
    
    # Summary statistics
    summary = calculate_summary_statistics(cleaned_df)
    print(summary)

    # Data filtering
    if args.filter_column and args.filter_value:
        filtered_df = filter_data(cleaned_df, args.filter_column, args.filter_value)
        # Save the filtered data
        save_processed_data(filtered_df, args.output_path)
    else:
        filtered_df = cleaned_df
    
    # Visualizations
    if args.histogram_col:
        plot_histogram(cleaned_df, args.histogram_col)
    if args.bar_category_col and args.bar_value_col:
        plot_bar_chart(cleaned_df, args.bar_category_col, args.bar_value_col)
    if args.pie_category_col and args.pie_value_col:
        plot_pie_chart(cleaned_df, args.pie_category_col, args.pie_value_col, args.pie_explode_index)
    if args.scatter_x_col and args.scatter_y_col:
        plot_scatter(cleaned_df, args.scatter_x_col, args.scatter_y_col, args.scatter_hue_col)

if __name__ == "__main__":
    main()