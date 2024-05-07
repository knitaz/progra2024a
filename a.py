import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Define the path to the directory containing the text files
CORPUS_DIR = 'data folder'

# Function to read the text files in the specified directory
def read_corpus(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents[filename] = file.read().lower()  # Read and convert to lowercase
    return documents

# Read the corpus
documents = read_corpus(CORPUS_DIR)

# Initialize the vectorizer to count the words
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")  # Use a custom token pattern

# Tokenize and count the words in all documents
matrix_count = vectorizer.fit_transform(documents.values())

# Get the unique words
words = vectorizer.get_feature_names_out()

# Convert the count matrix to a pandas DataFrame
df_matrix = pd.DataFrame(matrix_count.toarray(), index=words, columns=documents.keys())

print(df_matrix)
