import csv
import re
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

reviews = [row for row in csv.reader(open('victories.csv'))]


def process_text(text):
    # Make all the strings lowercase and remove non alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text.lower())

    # Tokenize the text; this is, separate every sentence into a list of words
    # Since the text is already split into sentences you don't have to call sent_tokenize
    tokenized_text = word_tokenize(text)

    # remove the stopwords and stem each word to its root
    clean_text = [
        stemmer.stem(word) for word in tokenized_text
        if word not in stopwords.words('english')
    ]

    return clean_text


# remove the first row, since it only has labels
reviews = reviews[1:]
texts = [row[0] for row in reviews]
topics = [row[2] for row in reviews]

# process the texts to so they are ready fro training
# but transform the list of words back to string format to feed it to sklearn
texts = [" ".join(process_text(text)) for text in texts]

print(reviews)

