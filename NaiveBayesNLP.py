from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import string

stopwords_english = stopwords.words('english')

# clean words, i.e. remove stopwords and punctuation
def clean_words(words, stopwords_english):
    words_clean = []
    for word in words:
        word = word.lower()
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)
    return words_clean

# feature extractor function for unigram
def bag_of_words(words):
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary

# feature extractor function for ngrams (bigram)
def bag_of_ngrams(words, n=2):
    words_ng = []
    for item in iter(ngrams(words, n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)
    return words_dictionary


text = "It was a very good movie."
words = word_tokenize(text.lower())

print (words)
print (bag_of_ngrams(words))

# working with cleaning words
# i.e. removing stopwords and punctuation
words_clean = clean_words(words, stopwords_english)
print (words_clean)

# cleaning words is find for unigrams
# but this can omit important words for bigrams
# for example, stopwords like very, over, under, so, etc. are important for bigrams
# we create a new stopwords list specifically for bigrams by omitting such important words
important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too', 'very', 'just', 'but']

stopwords_english_for_bigrams = set(stopwords_english) - set(important_words)

words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)
print (words_clean_for_bigrams)

# We will use general stopwords for unigrams
# And special stopwords list for bigrams
unigram_features = bag_of_words(words_clean)
print (unigram_features)

bigram_features = bag_of_ngrams(words_clean_for_bigrams)
print (bigram_features)

# combine both unigram and bigram features
all_features = unigram_features.copy()
all_features.update(bigram_features)
print (all_features)


# let's define a new function that extracts all features
# i.e. that extracts both unigram and bigrams features
def bag_of_all_words(words, n=2):
    words_clean = clean_words(words, stopwords_english)
    words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)

    unigram_features = bag_of_words(words_clean)
    bigram_features = bag_of_ngrams(words_clean_for_bigrams)

    all_features = unigram_features.copy()
    all_features.update(bigram_features)

    return all_features

print (bag_of_all_words(words))

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)

neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)

# positive reviews feature set
pos_reviews_set = []
for words in pos_reviews:
    pos_reviews_set.append((bag_of_all_words(words), 'pos'))

# negative reviews feature set
neg_reviews_set = []
for words in neg_reviews:
    neg_reviews_set.append((bag_of_all_words(words), 'neg'))

    print (len(pos_reviews_set), len(neg_reviews_set)) # Output: (1000, 1000)

# randomize pos_reviews_set and neg_reviews_set
# doing so will output different accuracy result every time we run the program
from random import shuffle
shuffle(pos_reviews_set)
shuffle(neg_reviews_set)

test_set = pos_reviews_set[:200] + neg_reviews_set[:200]
train_set = pos_reviews_set[200:] + neg_reviews_set[200:]

print(len(test_set),  len(train_set))

classifier = NaiveBayesClassifier.train(train_set)

accuracy = classify.accuracy(classifier, test_set)
print(accuracy) # Output: 0.8025

print (classifier.show_most_informative_features(10))

custom_review = "I hated the film. It was a disaster. Poor direction, bad acting."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_all_words(custom_review_tokens)
print (classifier.classify(custom_review_set)) # Output: neg
# Negative review correctly classified as negative

# probability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result) # Output: <ProbDist with 2 samples>
print (prob_result.max()) # Output: neg
print (prob_result.prob("neg")) # Output: 0.770612685688
print (prob_result.prob("pos")) # Output: 0.229387314312


custom_review = "It was a wonderful and amazing movie. I loved it. Best direction, good acting."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_all_words(custom_review_tokens)

print (classifier.classify(custom_review_set)) # Output: pos
# Positive review correctly classified as positive

# probability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result) # Output: <ProbDist with 2 samples>
print (prob_result.max()) # Output: pos
print (prob_result.prob("neg")) # Output: 0.00677736186354
print (prob_result.prob("pos")) # Output: 0.993222638136
