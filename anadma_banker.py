import numpy as np
from sklearn.naive_bayes import MultinomialNB

np.seterr(all='ignore')
class AnadmaBanker:
    """
    A class for an algorithm to predict whether a loan
    should be granted.

    Methods:
    fit(X, y, alpha=1.0)        : Fits the classifier to the data
    set_interest_rate(rate)     : Sets the interest rate
    predict_proba(x)            : Predicts whether a person will repay the loan
    expected_utility(x, action) : Finds the expected utility of granting a loan
    get_best_action(x)          : Returns the suggested action
    """

    def fit(self, X, y, alpha=1.0):
        """
        A function to fit a Multinomial Naive Bayes classifier to data.

        Parameters:
        X       (Pandas dataframe): The data without labels
        y       (Pandas dataframe): The labels of the data
        alpha   (float, optional): The additive smoothing parameter

        Returns:
        clf (MultinomialNB classifier): The classifier fitted to the data 
        """

        self.data = [X, y]
        self.clf = MultinomialNB(alpha=alpha)
        self.clf.fit(X, y)
        return self.clf
       
    def set_interest_rate(self, rate):
        """
        A function to set the interest rate for the classifier

        Parameters:
        rate (float): The interest rate

        Returns:

        """

        self.rate = rate
        return

    def predict_proba(self, x):
        """
        Predicts whether a person will repay the loan or not.

        Parameters:
        x (Pandas dataframe): The person we want to make our prediction on

        Returns:
        predict[0][0] (float): The proability of a person repaying
        """

        x = [x]
        predict = self.clf.predict_proba(x)
        return predict[0][0]

    def expected_utility(self, x, action):
        """
        A function to calculate the expected utility for a person, given an action.

        Parameters:
        x       (Pandas dataframe): The person we want to make our prediction on
        action  (int): The action for calculation. 1 for grant loan, 0 for do not grant loan

        Returns:
        (float): The expected utility for the given action
        """

        if action == 0:
            return 0
        else:
            predict = self.predict_proba(x)
            return predict * x[4] * (np.power(1 + self.rate, x[1]) - 1) - (1 - predict)*x[4]

    def get_best_action(self, x):
        """
        A function to calculate the best action for a given person.

        Parameters:
        x (Pandas dataframe): The person we want to find the best action for

        Returns:
        (int): 1 if we should grant the loan, 0 if not
        """

        if self.expected_utility(x, 1) > self.expected_utility(x, 0):
            return 1
        else:
            return 0