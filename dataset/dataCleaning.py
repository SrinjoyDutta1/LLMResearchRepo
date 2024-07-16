
import pandas as pd
import re
from bs4 import BeautifulSoup


def clean_html(text):
    """Remove HTML tags and convert HTML entities to normal text."""
    return BeautifulSoup(text, "html.parser").get_text()


def contains_link(text):
    """Check if the text contains any hyperlink."""
    if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text):
        return True
    return False


def filter_and_merge(questions_path, answers_path, output_path):
    # Load the datasets
    try:
        questions_df = pd.read_csv(questions_path, parse_dates=['CreationDate', 'ClosedDate'])
        answers_df = pd.read_csv(answers_path, parse_dates=['CreationDate'])
    except UnicodeDecodeError:
        questions_df = pd.read_csv(questions_path, encoding='ISO-8859-1', parse_dates=['CreationDate', 'ClosedDate'])
        answers_df = pd.read_csv(answers_path, encoding='ISO-8859-1', parse_dates=['CreationDate'])

    # Filter answers with Score greater than 50
    answers_df = answers_df[answers_df['Score'] > 50]

    # Convert 'ClosedDate' where 'NA' values are present
    questions_df['ClosedDate'] = pd.to_datetime(questions_df['ClosedDate'], errors='coerce')

    # Clean HTML from 'Body' columns in both datasets
    questions_df['Body'] = questions_df['Body'].apply(clean_html)
    answers_df['Body'] = answers_df['Body'].apply(clean_html)

    # Filter out entries with links in the Body
    questions_df = questions_df[~questions_df['Body'].apply(contains_link)]
    answers_df = answers_df[~answers_df['Body'].apply(contains_link)]

    # Merging Answers with Questions using ParentId from Answers and Id from Questions
    merged_df = answers_df.merge(questions_df, left_on='ParentId', right_on='Id', suffixes=('_answer', '_question'))

    # Save the cleaned and merged data to a new CSV file
    merged_df.to_csv(output_path, index=False)
    print(f"Filtered and merged data saved to '{output_path}'.")

    # Display the first few rows of the merged data to verify
    print(merged_df.head())

#find keywords for these and use them for each topic to filter based on the topic you want.
def filter_information_security(input_path, output_path):
    keywords = ['information security', 'cybersecurity', 'data privacy', 'safety', 'ethics', 'ethical']

    def contains_keywords(text, keywords):
        """Check if the text contains any of the specified keywords."""
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                return True
        return False

    # Load the filtered merged dataset
    df = pd.read_csv(input_path)

    # Filter questions based on keywords
    filtered_df = df[df['Body_question'].apply(lambda x: contains_keywords(x, keywords))]

    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_path, index=False)
    print(f"Information security questions saved to '{output_path}'.")

    # Display the first few rows of the filtered data to verify
    print(filtered_df.head())


def filter_by_date(input_path, output_path_before, output_path_after, cutoff_date):
    # Load the filtered merged dataset
    df = pd.read_csv(input_path, parse_dates=['CreationDate_question'])

    # Filter questions before the cutoff date
    df_before = df[df['CreationDate_question'] < cutoff_date]

    # Filter questions after the cutoff date
    df_after = df[df['CreationDate_question'] >= cutoff_date]

    # Save the filtered data to new CSV files
    df_before.to_csv(output_path_before, index=False)
    df_after.to_csv(output_path_after, index=False)
    print(f"Questions before {cutoff_date} saved to '{output_path_before}'.")
    print(f"Questions on or after {cutoff_date} saved to '{output_path_after}'.")

    # Display the first few rows of the filtered data to verify
    print(df_before.head())
    print(df_after.head())


if __name__ == "__main__":
    questions_path = 'LLMResearch/dataset/Questions.csv'
    answers_path = 'LLMResearch/dataset/Answers.csv'

    # Step 1: Filter and merge data
    filter_and_merge(questions_path, answers_path, 'filtered_merged.csv')

    # Step 2: Filter information security questions
    filter_information_security('filtered_merged.csv', 'informationsecurityquestions.csv')

    # Step 3: Filter by dates
    cutoff_date = pd.Timestamp('2022-11-30')
    filter_by_date('filtered_merged.csv', 'questions_before_chatgpt.csv', 'questions_after_chatgpt.csv', cutoff_date)


