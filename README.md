**Chatbot Using ChatterBot and NLP**
==================================================
This project is a simple rule-based chatbot implemented in Python using the ChatterBot library and Natural Language Processing (NLP) for text processing and response generation.

Project Overview
------------------------
The chatbot reads a text document (chatboat.txt) containing information on data science, tokenizes the sentences, and responds to user queries based on the 
content using TF-IDF vectorization and cosine similarity.

Features
----------------
1)Greets users.

2)Responds to queries based on content from chatboat.txt.

3)Uses lemmatization for text normalization.

4)Applies TF-IDF vectorization and cosine similarity to find relevant responses.

5)Supports exit commands like "bye" and responses to "thanks."

**Step-by-Step Setup**
================================
Prerequisites
-----------------------------
Ensure you have Python 3.10 or higher installed. The following libraries are required:

   -nltk for natural language processing
   
   -ChatterBot and chatterbot_corpus for the chatbot framework 
   
   -scikit-learn for vectorization and similarity metrics

**Step 1: Clone the Repository**
=====================================
git clone https://github.com/yourusername/chatbot_project.git
      -cd chatbot_project

**Step 2: Install Dependencies**
========================================
You can install the required libraries using pip. Run the following commands:
--------------------------------------------------------------------------------
pip install chatterbot

pip install chatterbot_corpus

pip install nltk

pip install scikit-learn

**Step 3: Download NLTK Data**
===========================================
Before running the chatbot, download necessary NLTK corpora:
----------------------------------------------------------------
import nltk

nltk.download('punkt')

nltk.download('wordnet')

nltk.download('omw-1.4')

**Step 4: Create the Input Text File**
=================================================
Place the chatboat.txt file in the project folder, containing the data on which the chatbot will operate. Here’s an example of the structure:
--------------------------------------------------------------------------------------------------------------------------------------------------
Data science is an interdisciplinary field...

**Step 5: Run the Chatbot**
===================================
To start the chatbot, run the following command:
-----------------------------------------------------
python chatbot.py
The chatbot will start interacting with you. To exit, type "bye" or "thanks".

**Sample Code**
============================
Here is a basic structure of the chatbot:
----------------------------------------------
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import random

import string

# Load and preprocess the text
f = open('chatboat.txt', 'r', errors='ignore')
raw_doc = f.read().lower()
sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)


# Function for greetings
def greet(sentence):
    GREET_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
    GREET_RESPONSES = ["hi", "hey", "*nods*", "hello", "I'm glad you're talking to me!"]
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSES)


# Function for generating response
def response(user_response):
    robo_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response += "I'm sorry, I don't understand."
    else:
        robo_response += sent_tokens[idx]
    return robo_response

**Future Improvements**
=======================================
Expand the corpus to include a wider range of topics.
Implement machine learning techniques for improved responses.
Add a web interface using Flask or Streamlit
