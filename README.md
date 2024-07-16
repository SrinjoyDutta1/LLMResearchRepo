# LLMResearch
Research on correctness score of LLMS vs human responses on StackOverflow
Project Outline: Comparing Generative AI and Stack Overflow

Objective To compare the effectiveness, accuracy, and user satisfaction between Generative AI (including ChatGPT and Google Gemini) and Stack Overflow for software-related questions.
Data Sources Generative AI Responses: Obtain answers using APIs from ChatGPT and Google Gemini. Stack Overflow Posts: Extract questions and their accepted answers via database.
Time Frame Pre-Generative AI Launch: Analyze Stack Overflow data before November 30, 2022. Post-Generative AI Launch: Focus on questions posted after November 30, 2022, to identify potential shifts.
Question Selection Criteria Languages: Focus on Python and Java. Question Types: Include debugging, conceptual, and implementation questions. Votes and Acceptance: Select questions with higher upvotes and accepted answers for quality assurance.
Data Analysis Plan Qualitative Analysis: Depth and relevance of answers, readability, question understanding. Quantitative Analysis: Correctness rate, comprehensiveness score, readability index, user satisfaction (Likert scale), engagement metrics (Stack Overflow). Comparative Analysis: Between Generative AI and Stack Overflow, and pre/post-Generative AI launch data on Stack Overflow. Shift in Stack Overflow Usage: Trends in question frequency, engagement, and answer rates.



Dataset is from this kaggle link: https://www.kaggle.com/datasets/stackoverflow/stacksample

Purpose of each file:
Answers.csv:

dataCleaning.py

Filtered_Merged_Questions_Answers.csv

Questions.csv

Tags.csv

CreateDB.py

FilterDatasetByTopic.py

LLMGenerationComparison.py

sqlDump.txt

testBard.py

This project uses python and aws(sql rdbms) as the primary technologies
The following are imports which were needed,
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import pandas as pd
import pymysql
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI #this was for using chatGPT's LLM model
import google.generativeai as genai #this was for using google's LLM model
pip install nltk matplotlib seaborn scikit-learn gensim
pip install nltk matplotlib seaborn scikit-learn gensim
pip install transformers pandas scikit-learn torch
pip install imbalanced-learn
pip install accelerate -U
pip install -U google-generativeai
pip install SQLAlchemy



