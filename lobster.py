# -*- coding: utf-8 -*-
""" Module to used in multiprocess machine learning with nested cross validation.

Functions on this module are multilayer perceptron, and ...

Author:
    Marko Loponen
    majulop@utu.fi
    University of Turku

Versions and dates:
    6.2.2020: Initial and added multilayer perceptron function.

Todo:
    * Add kNN and Ridge Regression

"""

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import LeaveOneOut
""" Importing needed modules for this module
"""

def mlp(lsize, X_train, y_train, X_test, y_test):
    """ Multilayer Perceptron classifier

    Trains the model and predicts with the test set.

    Args:
        lsize (int): Neuron size.
        X_train (Pandas.DataFrame): Train set features.
        y_train (Pandas.DataFrame): Train set labels.
        X_test (Pandas.DataFrame): Test set features.
        y_test (Pandas.DataFrame): Test set labels

    Returns:
        tuple:  Tuple with value of Hidden layer size (int) and it's accuracy (float).
                Example: (3, 0.64302)
            
    """
    
    inner_predict_corr = 0
    """ Counter for how many was predicted correctly in the inner loop
    """
    
    loo_inner = LeaveOneOut()
    """ Leave one out
    """

    for inner_train_index, inner_test_index in loo_inner.split(X_train):
        
        inner_X_train, inner_X_test = X_train.iloc[inner_train_index,:], X_train.iloc[inner_test_index,:]
        inner_y_train, inner_y_test = y_train.iloc[inner_train_index,:], y_train.iloc[inner_test_index,:]
        """ Splitting to train and test sets
        """

        mlp = MLPClassifier(solver='adam', activation='relu', validation_fraction=0.5, early_stopping=True, alpha=1e-5, hidden_layer_sizes=(lsize,), random_state=1)
        mlp.fit(inner_X_train, inner_y_train.values.ravel())
        """ Setting Multilayer Perceptron classifier and fitting the inner train sets to it
        """
        
        if mlp.predict(inner_X_test)[0] == inner_y_test.iloc[0][0]:
            inner_predict_corr += 1
        """ Testing did it predict right and incrementing inner predict correct variable by one in that case
        """

    return (str(lsize), inner_predict_corr / X_train.shape[0])

def ridge(a, X_train, y_train, X_test, y_test):
    """ Ridge Regression classifier

    Trains the model and predicts with the test set.

    Args:
        a (int): Alpha value.
        X_train (Pandas.DataFrame): Train set features.
        y_train (Pandas.DataFrame): Train set labels.
        X_test (Pandas.DataFrame): Test set features.
        y_test (Pandas.DataFrame): Test set labels

    Returns:
        tuple:  Tuple with value of Alpha (float) and it's accuracy (float).
                Example: (0.2, 0.64302)
            
    """
    
    inner_predict_corr = 0
    """ Counter for how many was predicted correctly in the inner loop
    """

    loo_inner = LeaveOneOut()
    """ Leave one out
    """

    for inner_train_index, inner_test_index in loo_inner.split(X_train):

        inner_X_train, inner_X_test = X_train.iloc[inner_train_index,:], X_train.iloc[inner_test_index,:]
        inner_y_train, inner_y_test = y_train[inner_train_index], y_train[inner_test_index]
        """ Splitting the inner fold to train and test sets
        """

        clf = RidgeClassifier(alpha=a).fit(inner_X_train, inner_y_train)
        """ Setting Ridge classifier and fitting the inner train sets to it
        """

        if clf.predict(inner_X_test) == inner_y_test[0]:
            inner_predict_corr += 1
        """ Testing did it predict right and incrementing inner predict correct variable by one in that case
        """
        
    return (str(a), inner_predict_corr / X_train.shape[0])

def knn(k, X_train, y_train, X_test, y_test):
    """ kNN classifier

    Trains the model and predicts with the test set.

    Args:
        k (int): k-nearest neighbours.
        X_train (Pandas.DataFrame): Train set features.
        y_train (Pandas.DataFrame): Train set labels.
        X_test (Pandas.DataFrame): Test set features.
        y_test (Pandas.DataFrame): Test set labels

    Returns:
        tuple:  Tuple with value of k (int) and it's accuracy (float).
                Example: (3, 0.64302)
            
    """
    
    inner_predict_corr = 0
    """ Counter for how many was predicted correctly in the inner loop
    """

    loo_inner = LeaveOneOut()
    """ Leave one out
    """

    for inner_train_index, inner_test_index in loo_inner.split(X_train):

        inner_X_train, inner_X_test = X_train.iloc[inner_train_index,:], X_train.iloc[inner_test_index,:]
        inner_y_train, inner_y_test = y_train.iloc[inner_train_index,:], y_train.iloc[inner_test_index,:]
        """ Splitting the inner fold to train and test sets
        """

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(inner_X_train, inner_y_train.values.ravel())
        """ Setting kNN classifier and fitting the train sets to it
        """

        if knn.predict(inner_X_test) == inner_y_test['category'].iloc[0]:
            inner_predict_corr += 1
        """ Testing did it predict right and incrementing inner predict correct variable by one in that case
        """
        
    return (str(k), inner_predict_corr / X_train.shape[0])