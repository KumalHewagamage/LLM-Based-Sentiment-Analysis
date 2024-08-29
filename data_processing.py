
import pandas as pd
from Emoticon import EMOTICONS_EMO
import seaborn as sns
import matplotlib.pyplot as plt

def create_sentiment(row):
    """Convert rating to sentiment."""
    return 1 if row >= 4 else 0

def convert_emoticons_to_words(row):
    """Convert emojis in comments into text."""
    for i, j in EMOTICONS_EMO.items():
        row = row.replace(i, j)
    return row

def load_and_clean_data(file_path):
    """Load and clean the dataset."""
    df = pd.read_csv(file_path, low_memory=False)
    df = df[(df['reviews.text'].notnull()) & (df['reviews.rating'].notnull())]
    df['label'] = df['reviews.rating'].apply(create_sentiment)
    df = df[['reviews.text', 'label']].rename(columns={'reviews.text': "text"})
    df = df.drop_duplicates(subset=['text'], keep='last')
    df['text'] = df['text'].apply(convert_emoticons_to_words)
    return df

def visualize_class_distribution(data):
    """Visualize the class distribution of sentiments."""
    colors = ['royalblue', 'pink']
    fig, ax = plt.subplots()
    ax.pie(data['label'].value_counts(), labels=["positive", "negative"], autopct='%1.1f%%', colors=colors)
    plt.title("Sentiment Class Distribution")
    plt.show()
