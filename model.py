from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import pickle
import nltk.classify.util
import sys
import matplotlib.pyplot as plt
import numpy as np
nltk.download('movie_reviews')

def extract(words):
    return dict([(word, True) for word in words])

negative_ids = movie_reviews.fileids('neg')
positive_ids = movie_reviews.fileids('pos')


negative_features = [(extract(movie_reviews.words(fileids=[f])), 'neg') for f in negative_ids]
positive_features = [(extract(movie_reviews.words(fileids=[f])), 'pos') for f in positive_ids]


negative_cutoff = int(len(negative_features) * 99/100)
positive_cutoff = int(len(positive_features) * 45/50)

train_features = negative_features[:negative_cutoff] + positive_features[:positive_cutoff]
test_features = negative_features[negative_cutoff:] + positive_features[positive_cutoff:]

print('Training on %d instances, testing on %d instances' % (len(train_features), len(test_features)))
classifier = NaiveBayesClassifier.train(train_features)
print('Training complete')
print('accuracy:', nltk.classify.util.accuracy(classifier, test_features)*100,'%')
classifier.show_most_informative_features()


f = open('classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()
