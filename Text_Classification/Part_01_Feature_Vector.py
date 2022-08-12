# Bag-of-words (BOW) model is a common way in NLP to create feature vectors out of text.
# According to this model we use  CountVectorizer (sklearn) to create a feature vectors
# These vectors will be used for a machine learning model

from sklearn.feature_extraction.text import CountVectorizer

# corpus is the raw data list
sentences_corpus = ['John likes ice cream', 'John hates chocolate.']
print("Corupus: ", sentences_corpus)

# vocabulary consists of all five words in the sentences (corpus)
# key: the word, value: arbitrary index (I think it's arbitrary)
sentences_vocabulary = CountVectorizer(min_df=0, lowercase=False)
sentences_vocabulary.fit(sentences_corpus)
print("Vocabulary (or Vectorizer): ", sentences_vocabulary.vocabulary_)

# The vector is a list, where every element is a list of a sentence
# The index of every element of the inner list is one of the values of the dict (vocabulary)
# The valus of every element of the inner list is 0 or 1 and represent the occurrences of the key of the dict (vocabulary) in the sentence
# intuition: Both sentences have "John" so the value 1 in the first element of the Feature vector in both of them represent they have the same first
sentences_feature_vector = sentences_vocabulary.transform(sentences_corpus).toarray()
print("Featutre Vector: ", sentences_feature_vector)

