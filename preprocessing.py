import pandas as pd
from stop_words import get_stop_words
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')


papers = pd.read_csv('victories.csv', encoding='windows-1252')

papers['paper_text_processed'] = papers['Petition Title'].map(lambda x: re.sub('[,\.!?]', '', x))

papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())

# Join the different processed titles together.
long_string = ','.join(list(papers['paper_text_processed'].values))

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
cloud=wordcloud.generate(long_string)

# Visualize the word cloud
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


data = papers.paper_text_processed.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])


