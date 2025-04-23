import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load English stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the spaCy English model
import en_core_web_sm
nlp = en_core_web_sm.load()

# Define text preprocessing function
def preprocess_text(text):
    # 1. Lowercase the text
    text = text.lower()
    # 2. Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # 3. Tokenize the text
    tokens = word_tokenize(text)
    # 4. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # 5. Lemmatize the tokens
    doc = nlp(' '.join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]

    # Return the cleaned text
    return ' '.join(lemmatized_tokens)

# Main processing function
def load_and_clean_data(filepath="suicide.csv", save_cleaned=True):
    df = pd.read_csv(filepath)
    df['clean_text'] = df['text'].apply(preprocess_text)

    if save_cleaned:
        df.to_csv("cleaned_suicide_dataset.csv", index=False)
        print("✅ Cleaned dataset saved to 'cleaned_suicide_dataset.csv'")
    return df

# For testing this file directly
if __name__ == "__main__":
    df = load_and_clean_data()
    print(df[['text', 'clean_text']].head())
