#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:47:44 2024

@author: bobnguyen
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
nltk.download('averaged_perceptron_tagger')  
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np   
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('Books_rating.csv')
df2 = pd.read_csv('books_data.csv')
df = df.merge(df2[['Title', 'categories']], on='Title', how='left')
print(df.shape)
df.head()

sample = df.head(5000)

sample.head()
##Removing the NA values
sample.isnull().sum()
sample.dropna(how='any',inplace=True)

##Removing the Duplicated values
sample.duplicated().sum()
sample.drop_duplicates(inplace=True)

selected_columns = ['Id','Title','User_id','review/text','categories']
sample_selected_columns = sample[selected_columns].drop_duplicates(subset='User_id', keep='first')
sample_selected_columns.head(5)

def preprocess_text(text):
    # Lowercase
    text = text.lower()   
    # Removal of URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove alphanumeric words
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Part-of-speech tagging
    tagged_tokens = nltk.pos_tag(tokens)
    # Filter tokens that are nouns or adjectives
    # NN* for nouns, JJ* for adjectives
    tokens = [word for word, tag in tagged_tokens if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    extra_words = {'.','*',','}
    stop_words = stop_words.union(extra_words)
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens
    

# Apply preprocess_text to each entry in the 'Text' column
sample_selected_columns['Processed_Text'] = sample_selected_columns['review/text'].apply(preprocess_text)
  

# 'texts' will be a list of lists of tokens
texts = sample_selected_columns['Processed_Text'].tolist()



# Create a dictionary
dictionary = Dictionary(texts)

# Filter out high-frequency terms
# Get the frequency of each term
term_freq = dictionary.cfs

# Define a threshold for high frequency (you can adjust this threshold)
# Here, we remove terms that appear in more than 20% of the documents
threshold = 0.3 * len(texts)
high_freq_terms = [tokenid for tokenid, freq in term_freq.items() if freq > threshold]

# Filter out high-frequency terms from the dictionary
dictionary.filter_tokens(high_freq_terms)

# Create a document-term matrix (corpus)
corpus = [dictionary.doc2bow(text) for text in texts]


# Set parameters
num_topics = 6
passes = 10

# Create the LDA model
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

# Display the topics
topics = lda.print_topics(num_words=10)
for topic in topics:
    print(topic)

import matplotlib.pyplot as plt

# Range of topics to evaluate
topic_range = range(2, 10)

# List to store coherence
coherence_scores = []

for num_topics in topic_range:
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=100)
    # Initialize CoherenceModel after training the LDA model
    coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    # Append the coherence score
    coherence_scores.append(coherence_lda)

print(coherence_scores)
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(topic_range, coherence_scores)
plt.title("Coherence Scores vs Number of Topics")
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.tight_layout()
plt.show()     


# Print the top 5 terms for each topic
topic_names = []
for topic_id in range(num_topics):
    top_terms = lda.show_topic(topic_id, topn=5)  # Get the top 5 terms for this topic
    terms = ', '.join([term for term, _ in top_terms])
    print(f"Topic {topic_id}: {terms}")
    
# Here you can define topic names based on the terms
    if topic_id == 0:
        topic_names.append("Parenting Gifted Children")
    elif topic_id == 1:
        topic_names.append("Financial Management and Religious Figures")
    elif topic_id == 2:
        topic_names.append("Philosophies of Thomas Paine")
    elif topic_id == 3:
        topic_names.append("Family Dynamics and Pets")
    elif topic_id == 4:
        topic_names.append("Home Financing and Education")
    elif topic_id == 5:
        topic_names.append("Children, Paganism, and Life Reviews")

# Assign each review to the most likely topic
def assign_topic(review, dictionary, lda_model):
    bow = dictionary.doc2bow(review)
    topics = lda_model.get_document_topics(bow)
    return max(topics, key=lambda x: x[1])[0]

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

train['Assigned_Topic'] = train['Processed_Text'].apply(lambda x: assign_topic(x, dictionary, lda))

# Profile categories purchased by users in each topic
topic_profiles = train.groupby('Assigned_Topic')['categories'].value_counts().groupby(level=0).nlargest(5).reset_index(level=0, drop=True)
topic_profiles = topic_profiles.groupby(level=0).apply(list).to_dict()

# Normalize the user-topic distribution for clustering
user_topic_distribution_train = train.groupby('User_id')['Assigned_Topic'].apply(list)
user_topic_distribution_train = user_topic_distribution_train.apply(lambda x: np.bincount(x, minlength=num_topics) / len(x))

# Assign topics to users in the test set
test['Assigned_Topic'] = test['Processed_Text'].apply(lambda x: assign_topic(x, dictionary, lda))

# Create user-item matrices for training and testing
train_user_item_matrix = train.pivot_table(index='User_id', columns='categories', aggfunc='size', fill_value=0)
test_user_item_matrix = test.pivot_table(index='User_id', columns='categories', aggfunc='size', fill_value=0)

# Calculate cosine similarity between users
user_similarities = cosine_similarity(user_topic_distribution_train.tolist())
user_similarities_df = pd.DataFrame(user_similarities, index=user_topic_distribution_train.index, columns=user_topic_distribution_train.index)

# Function to get top N similar users
def get_top_n_similar_users(user_id, n=5):
    if user_id not in user_similarities_df.index:
        return []
    similar_users = user_similarities_df[user_id].sort_values(ascending=False).head(n).index
    return similar_users

# Function to recommend categories for a user in the test set based on their assigned topic
def recommend_categories(user_id, n_recommendations=5):
    if user_id not in test['User_id'].values:
        print(f"User {user_id} not found in test data")
        return []
    
    assigned_topic = test.loc[test['User_id'] == user_id, 'Assigned_Topic'].values[0]
    top_similar_users = get_top_n_similar_users(user_id)
    if len(top_similar_users) == 0:
        print(f"No similar users found for {user_id}")
        return []
    
    similar_users_items = train_user_item_matrix.loc[top_similar_users].mean().sort_values(ascending=False)
    already_reviewed = test_user_item_matrix.loc[user_id][test_user_item_matrix.loc[user_id] > 0].index
    recommendations = similar_users_items.drop(already_reviewed, errors='ignore').head(n_recommendations).index
    return recommendations

# Example of getting recommendations for a user
user_id_example = 'User1'
recommendations = recommend_categories(user_id_example)
print(f"Recommendations for {user_id_example}: {recommendations}")

# Function to calculate precision, recall, and f1-score for a user
def evaluate_user(user_id, n_recommendations=5):
    if user_id not in test['User_id'].values:
        return 0, 0, 0
    
    recommendations = recommend_categories(user_id, n_recommendations)
    actual_items = test_user_item_matrix.loc[user_id][test_user_item_matrix.loc[user_id] > 0].index
    
    if len(actual_items) == 0:
        return 0, 0, 0
    
    true_positives = len(set(recommendations) & set(actual_items))
    precision = true_positives / len(recommendations) if len(recommendations) > 0 else 0
    recall = true_positives / len(actual_items) if len(actual_items) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

# Calculate precision, recall, and f1-score for all users in the test set
precisions, recalls, f1s = [], [], []

for user_id in test['User_id'].values:
    precision, recall, f1 = evaluate_user(user_id)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

# Calculate average precision, recall, and f1-score
average_precision = np.mean(precisions)
average_recall = np.mean(recalls)
average_f1 = np.mean(f1s)

print(f"Average Precision: {average_precision}")
print(f"Average Recall: {average_recall}")
print(f"Average F1-Score: {average_f1}")