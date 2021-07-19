# Stemming Libraries
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

# Gensim
from gensim import corpora, models
import gensim
import re

# To read csv file
import pandas as pd

# Plotting tools
import pyLDAvis.gensim as gensimvis
import pyLDAvis
from wordcloud import WordCloud
import matplotlib.pyplot as plt


tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# create sample documents
data = pd.read_csv('combined-csv-files.csv', encoding='windows-1252')

doc_read = data['Petition_Title']

# doc_b = pd.read_csv('popular_weekly.csv', encoding='windows-1252')
# doc_c = pd.read_csv('recent.csv', encoding='windows-1252')
# doc_d = pd.read_csv('victories.csv', encoding='windows-1252')
# doc_e = "Health professionals say that brocolli is good for your health."
# doc_read_1 = doc_b['Petition Title']
# doc_read_2 = doc_d['Petition Title']
# doc_read_3 = doc_c['Petition Title']
# data = [doc_read_1, doc_read_2, doc_read_3]


# compile sample documents into a list
doc_set=doc_read.tolist()

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
# Create dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
# Term Document Frequency
corpus = [dictionary.doc2bow(text) for text in texts]



# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20, chunksize=100)
print(ldamodel.print_topics())
vis=gensimvis.prepare(ldamodel, corpus, dictionary=ldamodel.id2word)
pyLDAvis.save_html(vis, 'Topicviss.html')

# Pre-processing
data['paper_text_processed'] = data['Petition_Title'].map(lambda x: re.sub('[,\.!?]', '', x))

data['paper_text_processed'] = data['paper_text_processed'].map(lambda x: x.lower())

# Join the different processed titles together.
long_string = ','.join(list(data['paper_text_processed'].values))

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
cloud=wordcloud.generate(long_string)

# Visualize the word cloud
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

for t in range(n_components):
        plt.figure()
        plt.imshow(WordCloud().fit_words(lda_top_words[t]))
        plt.axis("off")
        plt.title("Topic #" + str(t))
        plt.show()