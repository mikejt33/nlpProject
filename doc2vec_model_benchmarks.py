# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import collections
import seaborn as sns

target_names = ["Chicken", "Punk", "perp", "Garbage", "Scum", "Toilet",
                "Poop", "Yuck"]

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)

opts.print_top10, opts.print_report, opts.print_cm = True, True, True

if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

# #############################################################################
categories = ["Chicken", "Punk", "perp", "Garbage", "Scum", "Toilet",
              "Poop", "Yuck"]

with open('Dan Coats.txt', 'r+') as in_file:
    textDan = in_file.read()
    sentsDan = nltk.sent_tokenize(textDan)

pdDan = pd.DataFrame({'sentence': sentsDan})
pdDan['author'] = 'Dan Coats'

with open('James Mattis.txt', 'r+') as in_file:
    textMattis = in_file.read()
    sentsMattis = nltk.sent_tokenize(textMattis)

pdMattis = pd.DataFrame({'sentence': sentsMattis})
pdMattis['author'] = 'James Mattis'

with open('John Kelly.txt', 'r+') as in_file:
    textKelly = in_file.read()
    sentsKelly = nltk.sent_tokenize(textKelly)

pdKelly = pd.DataFrame({'sentence': sentsKelly})
pdKelly['author'] = 'John Kelly'

with open('Kevin Hassett.txt', 'r+') as in_file:
    textHassett = in_file.read()
    sentsHassett = nltk.sent_tokenize(textHassett)

pdHassett = pd.DataFrame({'sentence': sentsHassett})
pdHassett['author'] = 'Kevin Hassett'

with open('Kirstjen Nielsen.txt', 'r+') as in_file:
    textNielsen = in_file.read()
    sentsNielsen = nltk.sent_tokenize(textNielsen)

pdNielsen = pd.DataFrame({'sentence': sentsNielsen})
pdNielsen['author'] = 'Kirstjen Nielsen'

with open('Larry Kudlow.txt', 'r+') as in_file:
    textKudlow = in_file.read()
    sentsKudlow = nltk.sent_tokenize(textKudlow)

pdKudlow = pd.DataFrame({'sentence': sentsKudlow})
pdKudlow['author'] = 'Larry Kudlow'

with open('Mike Pence.txt', 'r+') as in_file:
    textPence = in_file.read()
    sentsPence = nltk.sent_tokenize(textPence)

pdPence = pd.DataFrame({'sentence': sentsPence})
pdPence['author'] = 'Mike Pence'

with open('Mike Pompeo.txt', 'r+') as in_file:
    textPompeo = in_file.read()
    sentsPompeo = nltk.sent_tokenize(textPompeo)

pdPompeo = pd.DataFrame({'sentence': sentsPompeo})
pdPompeo['author'] = 'Mike Pompeo'

train = pd.DataFrame()
train = pd.concat(
    [pdDan, pdMattis, pdKelly, pdHassett, pdNielsen, pdKudlow, pdPence,
     pdPompeo])

author_to_num = {'Dan Coats': "Chicken", 'James Mattis': "Punk",
                 'John Kelly': "perp", 'Kevin Hassett': "Garbage",
                 'Kirstjen Nielsen': "Scum", 'Larry Kudlow': "Toilet",
                 'Mike Pence': "Poop", 'Mike Pompeo': "Yuck"}

train["author"].replace(author_to_num, inplace=True)
author = train['author'].tolist()

text = train['sentence'].tolist()
"""
get rid of non-breaking spaces,
double spaces,
NEW LINES
other unicode
"""
text = [t.replace('\xa0', ' ').replace("\u2028", " ").replace(u'\ufeff', ' ') \
            .replace('\n', " ").replace("  ", " ") for t in text]

print("there are ", len(text), "sentences")


def random_generator(size=6,
                     chars=list(string.ascii_uppercase + string.digits)):
    return ''.join(list(np.random.choice(chars, size)))


X_txt_train, X_txt_test, y_train, y_test = train_test_split(text, author,
                                                            test_size=0.25,
                                                            random_state=1337)
X_txt_train = [TaggedDocument(doc, [random_generator()]) for doc in X_txt_train]
X_txt_test = [TaggedDocument(doc, [random_generator()]) for doc in X_txt_test]

N_DIMS = 100
N_EPOCHS = 50

vectorizer = Doc2Vec(seed=1,
                     workers=multiprocessing.cpu_count(),
                     vector_size=N_DIMS,
                     dm=0,  # use distributed bag of words
                     min_count=0,
                     window=15,
                     epochs=N_EPOCHS)

vectorizer.build_vocab(X_txt_train)
vectorizer.train(X_txt_train, total_examples=vectorizer.corpus_count,
                 epochs=vectorizer.epochs)

X_train = [vectorizer.infer_vector(document.words) for document in X_txt_train]
# print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = [vectorizer.infer_vector(document.words) for document in X_txt_test]
duration = time() - t0


# print("n_samples: %d, n_features: %d" % X_test.shape)
# print()

# feature_names = vectorizer.get_feature_names()
#
# if opts.select_chi2:
#     print("Extracting %d best features by a chi-squared test" %
#           opts.select_chi2)
#     t0 = time()
#     ch2 = SelectKBest(chi2, k=opts.select_chi2)
#     X_train = ch2.fit_transform(X_train, y_train)
#     X_test = ch2.transform(X_test)
#     if feature_names:
#         # keep selected feature names
#         feature_names = [feature_names[i] for i
#                          in ch2.get_support(indices=True)]
#     print("done in %fs" % (time() - t0))
#     print()
#
# if feature_names:
#     feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    accuracy = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % accuracy)

    precision, recall, fscore, _support = metrics.precision_recall_fscore_support(
        y_test, pred,
        average="macro")
    # g

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    # if opts.print_cm:
    #     print("confusion matrix:")
    #     print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, accuracy, precision, recall, fscore, train_time, test_time


results = []
print('=' * 80)
print("Rand forest")
results.append(benchmark(RandomForestClassifier(n_estimators=100)))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

# Train sparse Naive Bayes classifiers
print('=' * 80)
# print("Naive Bayes")
# results.append(benchmark(MultinomialNB(alpha=.01)))
# results.append(benchmark(BernoulliNB(alpha=.01)))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(7)]

clf_names, accuracy, precision, recall, fscore, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Benchmarks for Various Classifiers using Doc2vec")
plt.barh(indices, accuracy, .2, label="Accuracy")
plt.barh(indices + .2, precision, .2, label="Precision")
plt.barh(indices + .4, recall, .2, label="Recall")
plt.barh(indices + .6, fscore, .2, label="F-Score")

plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-0.12, i, c)

# plt.savefig("doc2vec benchmarks.png")
plt.show()

## oped prediction
opEd = pd.read_table("OpEd.txt", header=None)
opEd.columns = ['text']

NYT_data = opEd['text'].tolist()
NYT_data = [TaggedDocument(doc, [random_generator()]) for doc in X_txt_test]
X_NYT = [vectorizer.infer_vector(document.words) for document in X_txt_train]

all_writing_samples = [TaggedDocument(doc, [random_generator()]) for doc in
                       text]
all_writing_samples_embedded = [vectorizer.infer_vector(document.words) for
                                document in all_writing_samples]

best_classifier = LinearSVC(penalty="l2", dual=False, tol=1e-3)
best_classifier.fit(all_writing_samples_embedded, author)

predictions = best_classifier.predict(X_NYT)

predictedAuthordf = pd.DataFrame(predictions)
predictedAuthordf.columns = ['Author']
predictedAuthordf = predictedAuthordf['Author'].value_counts().reset_index()
predictedAuthordf = pd.DataFrame(predictedAuthordf)
predictedAuthordf.columns = ['Author', 'Count']
predictedAuthordf["Probability"] = predictedAuthordf["Count"] / (
    predictedAuthordf['Count'].sum())
predictedAuthordf["logLikelihood"] = np.log(predictedAuthordf["Probability"])

prediction = predictedAuthordf['logLikelihood'].idxmax()
predictedAuthor = predictedAuthordf.at[prediction, 'Author']

print(predictedAuthordf, "\n\n")

# predictedAuthor = (list(possibleAuthors.keys())[list(possibleAuthors.values()).index(predictedAuthor)])
print("The predicted author is: ", predictedAuthor)

# print(predictions)
bins = len(set(predictions))
plt.hist(predictions, density=True, align="mid", bins=bins)
plt.xticks(range(bins))
plt.ylabel("% OpEd Sentences Attributed to Author", fontsize=16)
plt.xlabel("Author", fontsize=16)
plt.title("Percent of Sentences Attributed to the Authors", fontsize=20)
plt.show()
