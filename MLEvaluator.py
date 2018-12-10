import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
import scikitplot as skplt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier


def evaluate_features(X, y, clfKey):
    """
    General helper function for evaluating effectiveness of passed features in ML model

    Prints out Log loss, accuracy, and confusion matrix with 3-fold stratified cross-validation

    Parameters
    ----------
    X : Features array

    y : Labels array

    clfKey: 'LogReg', 'SDG_Mh', 'SDG_Log' and 'Tree'
    """

    clf = {
        'LogReg': LogisticRegression(),
        'SDG_Mh': SGDClassifier(loss="modified_huber", penalty="l2", max_iter=5),
        'SDG_Log': SGDClassifier(loss="log", penalty="l2", max_iter=5),
        'Tree': RandomForestClassifier(n_estimators=1000, max_depth=5, verbose=1)
    }

    probabilities = cross_val_predict(clf[clfKey], X, y, cv=StratifiedKFold(n_splits=2, random_state=8),
                                      n_jobs=-1, method='predict_proba', verbose=2)
    predicted_indices = np.argmax(probabilities, axis=1)
    classes = np.unique(y)
    predicted = classes[predicted_indices]
    print('Log loss: {}'.format(log_loss(y, probabilities)))
    print('Accuracy: {}'.format(accuracy_score(y, predicted)))
    skplt.metrics.plot_confusion_matrix(y, predicted, normalize=True)
    plt.show()


""" 
Information Input:
    This section reads in the Training information in and then attaches it to each other it will store it as a csv 
    and hold all of the information of a row together. 
"""
training_text = pd.read_csv('/Users/kingma/PycharmProjects/Final_AI_Project/training_text', sep='\|\|', header=None,
                            skiprows=1, names=["ID", "Text"], engine='python')

train_variants = pd.read_csv('/Users/kingma/PycharmProjects/Final_AI_Project/training_variants')

trainingDataSet = pd.merge(train_variants, training_text, how='left', on='ID')

# print(trainingDataSet.head())


"""
Evaluation/Build:
    
"""

VectorOneI = TfidfVectorizer(
    analyzer='word',
    strip_accents='unicode',
    stop_words='english',
    sublinear_tf=True
)

VectorOne = VectorOneI.fit_transform(trainingDataSet['Text'].values.astype('U'))

FeaturesOneI = TruncatedSVD(n_components=100, n_iter=5, random_state=12)

FeaturesOne = FeaturesOneI.fit_transform(VectorOne)

evaluate_features(
    FeaturesOne,
    trainingDataSet['Class'].values.ravel(),
    'LogReg'  # clfKey: 'LogReg', 'SDG_Log', 'SDG_Mh' and 'Tree'
)
