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
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
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

author_to_num ={'Dan Coats': "Chicken", 'James Mattis': "Punk",
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
text = [t.replace('\xa0', ' ').replace("\u2028", " ").replace(u'\ufeff',' ') \
        .replace('\n', " ").replace("  ", " ") for t in text]

print("there are ", len(text), "sentences")

X_txt_train, X_txt_test, y_train, y_test = train_test_split(text, author, test_size=0.25, random_state=1337)


# else:
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
X_train = vectorizer.fit_transform(X_txt_train)
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(X_txt_test)
duration = time() - t0
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

feature_names = np.asarray(vectorizer.get_feature_names())

def trim(s):
    return s
    # """Trim string to fit on terminal (assuming 80-column display)"""
    # return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
# Benchmark classifiers

scoring = ['precision_macro', 'recall_macro', "f1_macro", "accuracy"]

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    scores = cross_validate(clf, text, author, scoring=scoring,
                            cv = 5)

    # clf.fit(X_train, y_train)
    # train_time = time() - t0
    # print("train time: %0.3fs" % train_time)
    #
    # t0 = time()
    # pred = clf.predict(X_test)
    # test_time = time() - t0
    # print("test time:  %0.3fs" % test_time)
    # accuracy = metrics.accuracy_score(y_test, pred)
    # print("accuracy:   %0.3f" % accuracy)
    #
    # precision, recall, fscore, _support = metrics.precision_recall_fscore_support(y_test, pred,
    #                                            average="macro")
    #
    # if hasattr(clf, 'coef_'):
    #     print("coef shape: ", clf.coef_.shape)
    #     print("dimensionality: %d" % clf.coef_.shape[1])
    #
    #     if opts.print_top10 and feature_names is not None:
    #         print("top 10 keywords per class:")
    #         for i, label in enumerate(target_names):
    #             top10 = np.argsort(clf.coef_[i])[-10:]
    #             print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
    #     print()
    #
    #     if opts.print_top10 and feature_names is not None:
    #         print("bottom 10 keywords per class:")
    #         for i, label in enumerate(target_names):
    #             bottom10 = np.argsort(clf.coef_[i])[:10]
    #             print(trim("%s: %s" % (label, " ".join(feature_names[bottom10]))))
    #     print()
    #
    # if opts.print_report:
    #     print("classification report:")
    #     print(metrics.classification_report(y_test, pred,
    #                                         target_names=target_names))
    #
    # # if opts.print_cm:
    # #     print("confusion matrix:")
    # #     print(metrics.confusion_matrix(y_test, pred))
    #
    # print()
    # clf_descr = str(clf).split('(')[4]
    # return clf_descr, accuracy, precision, recall, fscore, train_time, test_time
    return scores


results = []

print('=' * 80)
print("L2 SVM")
pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('vectorize', TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')),
    ('classify', LinearSVC(penalty="l2", dual=False,
                                       tol=1e-3))
])
results.append(["SVM L2-Norm", benchmark(pipe)])
#
# print('=' * 80)
# print("Rand forest")
# results.append(benchmark(RandomForestClassifier(n_estimators=100)))
#
# for penalty in ["l2", "l1"]:
#     print('=' * 80)
#     print("%s penalty" % penalty.upper())
#     # Train Liblinear model
#     results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
#                                        tol=1e-3)))
#
#
# # Train sparse Naive Bayes classifiers
# print('=' * 80)
# print("Naive Bayes")
# results.append(benchmark(MultinomialNB(alpha=.01)))
# results.append(benchmark(BernoulliNB(alpha=.01)))


# make some plots

indices = np.arange(len(results))

#results = [[x[i] for x in results] for i in range(7)]

clf_names, scores = zip(*results)
scores_unpacked = []
for i, classifier in enumerate(clf_names):
     clf_score_dict = scores[i]
     for metric, array_ in clf_score_dict.items():
         unwanted = ['_time', "train_"]
         if not any(substring in metric for substring in unwanted):
             for value in array_:
                 print([classifier, metric, value])
                 scores_unpacked.append([str(classifier),
                                         str(metric),
                                         float(value)])

scores_df = pd.DataFrame(scores_unpacked,
                         columns=["classifier", "metric", "value"])gi

sns.boxplot(x="metric",
            y="value",
            hue="classifier",
            data=scores_df)
plt.show()

#
# clf_names, accuracy, precision, recall, fscore, training_time, test_time = results
# training_time = np.array(training_time) / np.max(training_time)
# test_time = np.array(test_time) / np.max(test_time)

# results = results[0]
# print(results["test_accuracy"])
#
# plt.figure(figsize=(12, 8))
# plt.title("Benchmarks for Various Classifiers using TF-iDF")
# plt.barh(indices, [score["test_accuracy"] for score in scores], .2, label="Accuracy", color='navy')
# plt.barh(indices + .2, [score["test_precision_macro"] for score in scores], .2, label="Precision")
# plt.barh(indices + .4, scores["test_recall_macro"], .2, label="Recall")
# plt.barh(indices + .6, scores["test_f1_macro"], .2, label="F-Score")
#
# plt.yticks(())
# plt.legend(loc='best')
# plt.subplots_adjust(left=.25)
# plt.subplots_adjust(top=.95)
# plt.subplots_adjust(bottom=.05)
#
# for i, c in zip(indices, clf_names):
#     plt.text(-.3, i, c)
#
# # plt.savefig("tf-idf benchmarks.png")
# plt.show()