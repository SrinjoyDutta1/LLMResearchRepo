import pymysql

def main():
    # Connect to the AWS RDS MySQL instance (without specifying the database)
  #enter ur own aws creds here
    db = pymysql.connect(
        host='',
        user='',
        password=''
    )

    # Create a cursor object
    cursor = db.cursor()

    # Create the LLMResearch database if it doesn't exist
    create_db_sql = "CREATE DATABASE IF NOT EXISTS LLMResearch;"
    cursor.execute(create_db_sql)

    # Select the LLMResearch database
    use_db_sql = "USE LLMResearch;"
    cursor.execute(use_db_sql)

    # Create the LLMComparison table if it doesn't exist
    create_table_sql = '''CREATE TABLE IF NOT EXISTS LLMComparison (
        ID INT PRIMARY KEY AUTO_INCREMENT,
        StackOverflowQuestionsNumLikes INT,
        StackOverflowAnswer TEXT,
        StackOverflowQuestion TEXT,
        ChatGPTResponse TEXT,
        GeminiResponse TEXT,
        BERTCosineScore FLOAT
    );'''
    cursor.execute(create_table_sql)

    # Commit the changes
    db.commit()

    # Show the tables in the LLMResearch database
    show_tables_sql = "SHOW TABLES;"
    cursor.execute(show_tables_sql)
    tables = cursor.fetchall()

    # Print the tables
    for table in tables:
        print(table)

    # Close the cursor and the connection
    cursor.close()
    db.close()

if __name__ == "__main__":
    main()
