import pymysql
import pandas as pd
import numpy as np
import os

def fetch_data_from_db():
    # Connect to the AWS RDS MySQL instance
    cnx = pymysql.connect(
        host='database-1.cbkiicwyecbs.us-east-1.rds.amazonaws.com',
        user='admin',
        password='LLMResearch1123!',
        database='LLMResearch'
    )
    
    query = "SELECT * FROM LLMComparison"
    df = pd.read_sql(query, cnx)
    
    cnx.close()
    
    return df

def save_to_csv(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def process_and_save_specific_columns(df):
    specific_columns = ['BERTCosineScore_ChatGPT', 'BERTCosineScore_Bard', 'Category', 'CurrentIndex']
    df_specific = df[specific_columns]
    
    save_to_csv(df_specific, '/Users/srinjoydutta/Desktop/LLMResearch/dataset/BertCosineScores.csv')

def calculate_averages(df):
    average_chatgpt = df['BERTCosineScore_ChatGPT'].mean()
    average_bard = df['BERTCosineScore_Bard'].mean()
    
    averages_by_category = df.groupby('Category')[['BERTCosineScore_ChatGPT', 'BERTCosineScore_Bard']].mean()
    
    return average_chatgpt, average_bard, averages_by_category

def additional_statistical_analysis(df):
    chatgpt_std = df['BERTCosineScore_ChatGPT'].std()
    bard_std = df['BERTCosineScore_Bard'].std()
    
    chatgpt_median = df['BERTCosineScore_ChatGPT'].median()
    bard_median = df['BERTCosineScore_Bard'].median()
    
    chatgpt_var = df['BERTCosineScore_ChatGPT'].var()
    bard_var = df['BERTCosineScore_Bard'].var()
    
    return {
        'chatgpt_std': chatgpt_std,
        'bard_std': bard_std,
        'chatgpt_median': chatgpt_median,
        'bard_median': bard_median,
        'chatgpt_var': chatgpt_var,
        'bard_var': bard_var
    }

def mean_and_std_by_category(df):
    mean_std_by_category = df.groupby('Category')[['BERTCosineScore_ChatGPT', 'BERTCosineScore_Bard']].agg(['mean', 'std'])
    return mean_std_by_category

def save_results_to_txt(average_chatgpt, average_bard, averages_by_category, stats, mean_std_by_category):
    filepath = '/Users/srinjoydutta/Desktop/LLMResearch/finalResults.txt'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(f"Average ChatGPT Score: {average_chatgpt}\n")
        f.write(f"Average Bard Score: {average_bard}\n")
        f.write("\nAverage Scores by Category:\n")
        f.write(averages_by_category.to_string())
        f.write("\n\nStatistical Analysis:\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n\nMean and Standard Deviation by Category:\n")
        f.write(mean_std_by_category.to_string())

def main():
    df = fetch_data_from_db()
    
    # Print the 'Category' column to the console
    print("Category Column:")
    print(df['Category'])
    
    
    # Save all data to finalResults.csv
    save_to_csv(df, '/Users/srinjoydutta/Desktop/LLMResearch/finalResults.csv')
    
    # Save specific columns to dataset/BertCosineScores.csv
    process_and_save_specific_columns(df)
    
    # Calculate averages
    average_chatgpt, average_bard, averages_by_category = calculate_averages(df)
    
    # Perform additional statistical analysis
    stats = additional_statistical_analysis(df)
    
    # Calculate mean and std by category
    mean_std_by_category = mean_and_std_by_category(df)
    
    # Save results to finalResults.txt
    save_results_to_txt(average_chatgpt, average_bard, averages_by_category, stats, mean_std_by_category)

if __name__ == "__main__":
    main()
