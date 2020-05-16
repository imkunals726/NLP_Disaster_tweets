from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.preprocessing import StandardScaler

from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer( 'english')

def num_remover( val):
    tokens = val.split()
    nums = [ str(i) for i in range(10)]
    final_tokens = []
    for token in tokens:
        token = token.strip()
        if not any( token.startswith( num ) for num in nums):
            final_tokens.append(token)
    return ' '.join(final_tokens)


def replace_urls(text):
    tokens = text.split()
    
    final_tokens = []
    
    for token in tokens:
        if token.lower().startswith('http'):
            final_tokens.append('url')
        elif token.lower().startswith('@'):
            final_tokens.append('taggeduser')
        else:
            final_tokens.append(token)
    return ' '.join(final_tokens)


def clean_text(df):

    replace_words = [ '&amp' , 'and' , '#' ]

    df['text'] = df['text'].apply(replace_urls)

    for word in replace_words :
        df[ 'text' ] = df[ 'text' ].str.replace( word , '' )

    df[ 'text' ] = df['text' ].apply( lambda txt : ' '.join( stemmer.stem(lemmatizer.lemmatize( word ) ) for word in txt.split( ' ') ) )

    df['keyword'] = df['keyword'].fillna('').str.replace('%20' , ' ')
    df[ 'text' ] = df.apply( lambda row : str( row[ 'text' ] ) + ' ' + str(row[ 'keyword' ]) if row[ 'keyword' ] else row[ 'text' ] , axis = 1)

    df['text'] = df['text'].apply(num_remover)
    
    return df

class MyDataTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.count_vectorizer = CountVectorizer()
        self.standard_scaler = StandardScaler()
    
    def fit(self, X, y=None):
        
        x = clean_text(X)
        x = self.count_vectorizer.fit_transform(X['text'])
        x = x.toarray()
        x = self.standard_scaler.fit(x)
        return self
        
    def transform(self, X, y=None):
        
        x = clean_text(X)
        x = self.count_vectorizer.transform(X['text'])
        x = x.toarray()
        x = self.standard_scaler.transform(x)
        x = x.T
        return x