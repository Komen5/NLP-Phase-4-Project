#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Import necessary libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Filter future Warnings
import warnings
warnings.filterwarnings("ignore")

# Data Preprocessing tools
from sklearn.preprocessing import MinMaxScaler

# Model Training and Evaluation
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from sklearn.pipeline import Pipeline

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score


# In[12]:


import pandas as pd

# Load the CSV file
df = pd.read_csv("data.csv", encoding="latin-1")


# In[13]:


import pandas as pd

def check_data(df):
    """
    Creates a DataFrame that checks for shape, size, info, and describe, and displays the output containing the information.
    
    Parameters:
    df (pd.DataFrame): The dataframe being described.
    
    Returns:
    None: Prints the description.
    """
    
    # Print shape of the DataFrame
    print("Shape of DataFrame:")
    print(df.shape)

    # Print size of the DataFrame
    print("Size of DataFrame:")
    print(df.size)

    # Print the info of the DataFrame
    print("Info of DataFrame:")
    print(df.info())

    # Print the descriptive statistics of the DataFrame
    print("Descriptive Statistics:")
    print(df.describe().T)


# In[14]:


import pandas as pd

def data_cleaning(df):
    """
    Cleans the data by checking for data types, column names, unique values, and missing values.
    
    Parameters:
    df (pd.DataFrame): The dataframe to be cleaned.
    
    Returns:
    None: Prints the data cleaning details.
    """
    
    # Print data types of the DataFrame
    print("Data Types of Columns:")
    print(df.dtypes)
    
    # Print column names of the DataFrame
    print("\nColumn Names:")
    print(df.columns)
    
    # Print unique values for each column
    print("\nUnique Values in Each Column:")
    for column in df.columns:
        print(f"{column}: {df[column].unique()}")
    
    # Print missing values for each column
    print("\nMissing Values in Each Column:")
    print(df.isnull().sum())


# In[15]:


import pandas as pd

def analyze_data(df):
    # Check for missing values in the DataFrame
    print("Missing Values:\n", df.isnull().sum())

    # Check for duplicates in the dataset
    print("\nDuplicate Rows in the dataset:\n", df.duplicated().sum())


# In[16]:


import re
import pandas as pd

def clean_tweet(tweet):
    """
    Cleans a tweet by removing:
    - Punctuation and special characters
    - Email addresses
    - Links (URLs)
    - Numbers
    - Mentions (@username)
    - Hashtags (#hashtag)
    - Converts the text to lowercase
    """
    # Convert tweet to lowercase
    tweet = tweet.lower()
    
    # Remove email addresses
    tweet = re.sub(r'\S+@\S+', '', tweet)
    
    # Remove links (http, https, or www)
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    
    # Remove mentions (e.g., @username)
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove hashtags but keep the text (e.g., #hashtag -> hashtag)
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    
    # Remove numbers
    tweet = re.sub(r'\d+', '', tweet)
    
    # Remove punctuation and special characters (leaving spaces)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    
    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    return tweet


# In[17]:


# After handling missing values, perform brand assignment based on keywords
apple_words = ['ipad', 'apple', 'iphone', 'itunes', 'ipad2']
google_words = ['google', 'android', 'youtube']

def assign_brand(row):
    tweet = row['tweet']
    brand = row['brand']
    
    if pd.notna(brand) and brand != 'other':
        return brand
    else:
        apple, google = False, False
        # look for apple keyword
        for word in apple_words:
            if word in tweet.lower():
                apple = True
                break
        # look for google keyword
        for word in google_words:
            if word in tweet.lower():
                google = True
                break
        
        # return correct new label
        if apple and not google:
            return 'apple'
        elif google and not apple:
            return 'google'
        elif apple and google:
            return 'both'
        else:
            return 'neither'


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

# Function to plot a bar chart for categorical columns
def plot_bar(df, column, title):
    '''Plots a bar chart for categorical columns'''
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(f'{title} Distribution')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Function to plot a distribution plot for numerical columns
def plot_distribution(df, column, title):
    '''Plots a distribution plot for numerical columns'''
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f'{title} Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Function to plot a distribution of tweet lengths
def plot_tweet_lengths(df):
    '''Plots a distribution of tweet lengths'''
    df['tweet_length'] = df['tweet'].apply(lambda x: len(str(x)))
    plot_distribution(df, 'tweet_length', 'Tweet Lengths')

# Function to print value counts summary for brand and emotion
def summary_value_counts(df):
    '''Prints the value counts for brand and emotion'''
    print("Brand Value Counts:")
    print(df['brand'].value_counts())
    print("\nEmotion Value Counts:")
    print(df['emotion'].value_counts())

# Function to generate and display a word cloud from tweet text
def generate_wordcloud(df):
    '''Generates and displays a word cloud from tweet text, with a specified font path'''
    
    # Path to the font file (for example, Arial on Windows)
    font_path = 'C:\\Windows\\Fonts\\arial.ttf'  # Update this path to the correct font file on your system
    
    text = ' '.join(df['tweet'])  # Join all tweet texts into one string
    
    # Create the word cloud, providing the font path
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Tweets')
    plt.show()


# Function to run the full univariate analysis
def run_univariate_analysis(df):
    '''Executes the full univariate analysis'''
    # Bar plots for brand and emotion
    plot_bar(df, 'brand', 'Brand')
    plot_bar(df, 'emotion', 'Emotion')
    
    # Tweet Length Distribution Plot
    plot_tweet_lengths(df)
    
    # Word Cloud
    generate_wordcloud(df)


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot a bivariate relationship between two categorical columns
def plot_categorical_relationship(df, col1, col2, title):
    '''Plots a bivariate relationship between two categorical columns using a countplot'''
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col1, hue=col2, data=df)
    plt.title(f'{title} - {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel('Count')
    plt.show()

# Function to plot the distribution of tweet length by emotion
def plot_tweet_length_by_emotion(df):
    '''Plots a boxplot to see how tweet lengths vary by emotion'''
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='emotion', y='tweet_length', data=df)
    plt.title('Tweet Length by Emotion')
    plt.xlabel('Emotion')
    plt.ylabel('Tweet Length')
    plt.show()

# Function to run the full bivariate analysis
def run_bivariate_analysis(df):
    '''Executes the full bivariate analysis'''
    # Bivariate analysis between brand and emotion
    plot_categorical_relationship(df, 'brand', 'emotion', 'Brand vs Emotion')

    # Tweet Length distribution by Emotion
    plot_tweet_length_by_emotion(df)


# In[20]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def tokenize(text):
    """
    Tokenizes text by:
    - Converting text to lowercase
    - Removing stop words
    - Removing words with less than 3 characters
    - Removing punctuation
    """
    # Tokenize the text and convert to lowercase
    tokens = word_tokenize(text.lower())
    
    # Remove stop words
    tokens = [t for t in tokens if t not in stop_words]
    
    # Remove punctuation and words with less than 3 characters
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2]
    
    return tokens


# In[ ]:




