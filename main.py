# Fake News Classifer with Naive Bayes
# Written by Pranav Eranki
# Dataset of approx. 250 articles
# 10/7/2018

from __future__ import division
from os import listdir
from os.path import isfile, join
import re, numpy as np, operator
from nltk.corpus import stopwords
from tqdm import *
import math, random

STOPWORDS = stopwords.words('english')
RAW_REAL = [open('data/Real/'+i, "r").read() for i in listdir('data/Real')] # Raw real data
RAW_FAKE = [open('data/Fake/'+i, "r").read() for i in listdir('data/Fake')] # Raw fake data

REAL = RAW_REAL[:int(.7*len(RAW_REAL))]
FAKE = RAW_FAKE[:int(.7*len(RAW_FAKE))]
TEST_REAL = RAW_REAL[-1*(len(RAW_REAL)-int(.7*len(RAW_REAL))):]
TEST_FAKE = RAW_FAKE[-1*(len(RAW_FAKE)-int(.7*len(RAW_FAKE))):]
ALL = REAL+FAKE

# Tokenization and pre-processing
def words(text): return re.findall(r'\w+', text.lower())
def tokenize(text): return [word for word in words(text) if word not in STOPWORDS]

REAL_WORDS = []
FAKE_WORDS = []
ALL_WORDS = []
for i in [tokenize(j) for j in REAL]:
        ALL_WORDS += i
        REAL_WORDS += i
for i in [tokenize(j) for j in FAKE]:
        ALL_WORDS += i
        FAKE_WORDS += i

P_REAL = len(REAL)/len(ALL)
P_FAKE = 1 - P_REAL

def P(word, cat): return cat.count(word) / len(cat)

def fake(word):
        if (word not in FAKE_WORDS):
                return 1
        P_w = P(word, FAKE_WORDS)
        NP_w = P(word, REAL_WORDS)
        return (P_w*P_FAKE)/(P_w*P_FAKE + NP_w*P_REAL+1e-10)

def real(word):
        if (word not in REAL_WORDS):
                return 1
        P_w = P(word, REAL_WORDS)
        NP_w = P(word, FAKE_WORDS)
        return (P_w*P_REAL)/(P_w*P_REAL + NP_w*P_FAKE+1e-10)

def sigmoid(x):
        return 1/(1+math.e**x)
def alpha(p):
        return math.log(1-p) - math.log(p)
def product(a):
        return reduce(operator.mul, a)

def predict(doc):
        #fake_likelihood = sigmoid(sum([alpha(fake(w)) for w in tokenize(doc)]))
        #real_likelihood = sigmoid(sum([alpha(real(w)) for w in tokenize(doc)]))
        fake_likelihood = product([fake(w) for w in tokenize(doc)])
        real_likelihood = product([real(w) for w in tokenize(doc)])
        prediction = np.argmax(np.asarray([fake_likelihood, real_likelihood]))
        return prediction

def test():
        count = 0
        false_p = 0 # Number reals classified as positive fakes
        false_n = 0 # Number of fakes negatively classified as reals
        for i in tqdm(range(len(TEST_REAL))):
                if predict(TEST_REAL[i]) == 1:
                        count += 1
                else:
                        false_p += 1

        for i in tqdm(range(len(TEST_FAKE))):
                if predict(TEST_FAKE[i]) == 0:
                        count += 1
                else:
                        false_n += 1

        # Assumes fakes are our positive
        print "False Negative Rate: " + str(false_n / len(TEST_FAKE))
        print "False Positive Rate: " + str(false_p / len(TEST_REAL))
        print "Accuracy: " + str(count / len(TEST_FAKE+TEST_REAL))
