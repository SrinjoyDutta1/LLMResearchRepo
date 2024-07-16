import os
import pandas as pd
import pymysql
import traceback
from sentence_transformers import SentenceTransformer, util
import openai
import google.generativeai as genai

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Initialize OpenAI and Google Generative AI clients
#enter your own api keys here
openai.api_key = ""
genai.configure(api_key="")

def clear_table(cnx):
    cursor = cnx.cursor()
    cursor.execute("DELETE FROM LLMComparison")
    cnx.commit()
    cursor.close()

def alter_table_if_needed(cnx):
    cursor = cnx.cursor()
    cursor.execute("SHOW COLUMNS FROM LLMComparison LIKE 'BERTCosineScore_ChatGPT'")
    result_chatgpt = cursor.fetchone()
    
    cursor.execute("SHOW COLUMNS FROM LLMComparison LIKE 'BERTCosineScore_Bard'")
    result_bard = cursor.fetchone()
    
    cursor.execute("SHOW COLUMNS FROM LLMComparison LIKE 'Category'")
    result_category = cursor.fetchone()

    cursor.execute("SHOW COLUMNS FROM LLMComparison LIKE 'CurrentIndex'")
    result_index = cursor.fetchone()
    
    if not result_chatgpt:
        cursor.execute("ALTER TABLE LLMComparison ADD COLUMN BERTCosineScore_ChatGPT FLOAT")
        
    if not result_bard:
        cursor.execute("ALTER TABLE LLMComparison ADD COLUMN BERTCosineScore_Bard FLOAT")
    
    if not result_category:
        cursor.execute("ALTER TABLE LLMComparison ADD COLUMN Category VARCHAR(255)")

    if not result_index:
        cursor.execute("ALTER TABLE LLMComparison ADD COLUMN CurrentIndex INT")
        
    cnx.commit()
    cursor.close()

def generateLLMResponses(cnx, df, start_index, end_index):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Load Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Slice the DataFrame based on start_index and end_index
    df_slice = df.iloc[start_index:end_index]

    last_processed_index = start_index

    for index, row in df_slice.iterrows():
        try:
            print("Likes\n", row['Score_answer'], "\nBODY-answer\n", row['Body_answer'], "\nBODYQuestion\n", row['Body_question'])
            likes = row['Score_answer']
            bodyAnswer = row['Body_answer']
            bodyQuestion = row['Body_question']
            category = row['Category']

            # Generate ChatGPT response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Answer the technical questions with the best of your knowledge."},
                    {"role": "user", "content": bodyQuestion},
                ]
            )
            chatGPTResponse = response.choices[0].message['content']

            # Generate Gemini response
            geminiModel = genai.GenerativeModel("gemini-1.5-flash")
            chat = geminiModel.start_chat(history=[])
            
            try:
                response = chat.send_message(bodyQuestion)
                BardAnswer = response.text
            except genai.types.generation_types.StopCandidateException as e:
                print(f"Error generating Gemini response: {e}")
                BardAnswer = "N/A"

            print('\n')
            print(f"GEMINI API RESPONSE: {BardAnswer}")
            print('\n')
            
            # Calculate semantic similarity using Sentence-BERT
            embeddings_answer = model.encode(bodyAnswer, convert_to_tensor=True)
            embeddings_chatGPT = model.encode(chatGPTResponse, convert_to_tensor=True)
            embeddings_bard = model.encode(BardAnswer, convert_to_tensor=True)

            cosine_similarity_chatGPT = util.cos_sim(embeddings_answer, embeddings_chatGPT).item()
            cosine_similarity_bard = util.cos_sim(embeddings_answer, embeddings_bard).item()

            print("Cosine Similarity (ChatGPT): ", cosine_similarity_chatGPT)
            print("Cosine Similarity (Bard): ", cosine_similarity_bard, "\n")

            # Insert into the database
            sql = """
            INSERT INTO LLMComparison (StackOverflowQuestionsNumLikes, StackOverflowAnswer, StackOverflowQuestion, ChatGPTResponse, GeminiResponse, BERTCosineScore_ChatGPT, BERTCosineScore_Bard, Category, CurrentIndex)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor = cnx.cursor()
            print("category is ", category)
            cursor.execute(sql, (likes, bodyAnswer, bodyQuestion, chatGPTResponse, BardAnswer, cosine_similarity_chatGPT, cosine_similarity_bard, category, index))
            cnx.commit()
            cursor.close()

            last_processed_index = index

        except Exception as e:
            print(f"Error at index {index}: {e}")
            traceback.print_exc()
            break

    return last_processed_index

def save_to_csv(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def main(start_index, end_index):
    # Load the CSV file
    file_path = 'LLMResearch/dataset/ClassifiedFilteredQuestions.csv'
    df = pd.read_csv(file_path)

    # Connect to the AWS RDS MySQL instance
  #enter your own cred here
    cnx = pymysql.connect(
        host='',
        user='',
        password='',
        database=''
    )
    
    # Clear the table
    #clear_table(cnx)
    
    alter_table_if_needed(cnx)
    last_index = generateLLMResponses(cnx, df, start_index, end_index)
    
    # Save the dataframe to CSV
    save_to_csv(df, 'LLMResearch/finalResults.csv')
    
    cnx.close()
    
    print(f"Last processed index: {last_index}")

if __name__ == "__main__":
    # Customize the range as needed
    main(0, 1500)  # For example, to process rows from 0 to 1500
