import pandas
import numpy as np
import matplotlib.pyplot as plt
import anadma_banker
import random_banker
from sklearn.model_selection import train_test_split
import scipy.stats as st

def mapping(x):
    """Map 2 to 0"""
    if x == 2:
        x = 0
    return x

def qua_noise(X):
    """
    Add noise to quantitative data.

    Takes a dataframe with quantitative data as 0 or 1 as input,
    adds either 0 or 1 to each datapoint with the probability of 0
    being 70%. Takes the new datapoint modulo 2, to return either
    0 or 1 as the new datapoint.

    Parameters:
    X   (Pandas dataframe): The dataframe you want to add noise to

    Returns:
    X   (Pandas dataframe): The dataframe with added noise
    """

    w = np.random.choice([0, 1], size=(len(X), len(quantitative_features_2)), p=[0.7, 0.3])
    X[quantitative_features_2] = (X[quantitative_features_2] + w) % 2
    return X

def laplace_func(X):
    """
    Adds noise to numerical data using laplace.

    Takes a dataframe with numerical data, and adds differential
    privacy using laplace. First we do a coin-toss to randomize
    what data gets noisy. We add noise based on the minimum and 
    maximum value of the data, and set all negative values to 0

    Parameters:
    X   (Pandas dataframe): The dataframe you want to add noise to

    Returns:
    X   (Pandas dataframe): The dataframe with added noise
    """

    X_noise = X.copy()
    epsilon = 5
    n = np.shape(X)[1]
    for i in numerical_features:
        if np.random.random() > 0.5:
            M = (X[i].max()-X[i].min())
            l = (M*epsilon)/n
            w = np.random.laplace(0, l)
            X_noise[i] += w
    X_noise[X_noise < 0] = 0
    return X_noise

def add_noise(X_train, X_test):
    """
    Function to apply noise to both quantitative and numerical data

    Parameters:
    X_train         (Pandas dataframe): The training data
    X_test          (Pandas dataframe): The test data

    Returns:
    X_train_noise   (Pandas dataframe): The training data with noise
    X_test_noise    (Pandas dataframe): The test data with noise
    """

    X_train_noise = laplace_func(X_train)
    X_test_noise = laplace_func(X_test)
    X_train_noise = qua_noise(X_train_noise)
    X_test_noise = qua_noise(X_test_noise)
    return X_train_noise, X_test_noise

def foreign(data):
    """
    A function to show the distribution of foreign workers in our data.

    Params:
    data    (Pandas dataframe): The data imported from german.data

    Returns:
    """

    counter = 0
    paid_back = 0
    for i in range(data.shape[0]):
        if data.iloc[i]['foreign'] == 'A202':
            counter += 1
            if data.iloc[i]['repaid'] == 1:
                paid_back += 1
    print("Number of foreign in dataset: ", counter)
    print("Number of those who paid back: ", paid_back)
    print("prosentage: ", paid_back/counter)
    print("total: ", 700/1000)

def women(data):
    """
    A function to show the distribution of women in our dataset.

    Parameters:
    data    (Pandas dataframe): The data imported from german.data

    Returns:
    """

    counter = 0
    paid_back = 0
    for i in range(data.shape[0]):
        if data.iloc[i]['marital status'] == 'A92' or data.iloc[i]['marital status'] == 'A95':
            counter += 1
            if data.iloc[i]['repaid'] == 1:
                paid_back += 1
    print("Number of women in dataset: ", counter)
    print("Number of women who paid back: ", paid_back)
    print("Percentage: ", paid_back/counter)
    print("Total: ", 700/1000)

def test_decision_maker(X_test, y_test, interest_rate, decision_maker):
    """
    A function to find the expect utility and return on investment for a decision maker.

    Takes a decision maker and compares the test data.
    If the decision maker grants a loan that would be repaid,
    we add that to total amount, and utility with an interest rate.
    If not, we subtract the amount from utility and add it to total.
    If we don't grant the loan, nothing happens.

    Parameters:
    X_test          (Pandas dataframe): The held out test set without labels
    y_test          (Pandas dataframe): The labels of the held out test set
    interest_rate   (float): The interest rate for the loan
    decision_maker  (object): A decision maker class trained on the german.data data

    Returns:
    utility (float):
    avg_roi (float): The average return on investment
    """
    
    #woman_not_loan, woman_loan, man_not_loan, man_loan
    n_test_examples = len(X_test)
    utility = 0

    total_amount = 0
    total_utility = 0
    decision_maker.set_interest_rate(interest_rate)
 
    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])
        good_loan = y_test.iloc[t]
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]

        if (action==1):
            if (good_loan != 1):
                utility -= amount
            else:    
                utility += amount*(pow(1 + interest_rate, duration) - 1)

        total_utility += utility
        total_amount += amount
    
    avg_roi = total_utility/total_amount

    return utility, avg_roi


features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'

df = pandas.read_csv('../../data/credit/german.data', sep=' ',
                     names=features+[target])
df['repaid'] = df['repaid'].map(mapping)

numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
quantitative_features_2 = []
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)

for i in X.columns:
    if i not in numerical_features and i != 'repaid':
        quantitative_features_2.append(i)

encoded_features = list(filter(lambda x: x != target, X.columns))

decision_maker = anadma_banker.AnadmaBanker()
rand_decision_maker = random_banker.RandomBanker()

interest_rate = 0.05
n_tests = 100
alp = 0.1
utility = 0
investment_return = 0
utility_list = []
invest_list = []

for iter in range(n_tests):
    X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], 
                                                        X[target], test_size=0.2)
    X_train_noise, X_test_noise = add_noise(X_train, X_test)

    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train_noise, y_train, alp)
    Ui, Ri, = test_decision_maker(X_test_noise, y_test, 
                                    interest_rate, decision_maker)
    utility += Ui
    investment_return += Ri
    utility_list.append(Ui)
    invest_list.append(Ri)

print("Average utility:", utility / n_tests)
print("95% confidence interval utility", st.t.interval(alpha=0.95, df=len(utility_list)-1, loc=np.mean(utility_list), scale=st.sem(utility_list)))

print("Average return on investment:", investment_return / n_tests)
print("95% confidence interval return on investment", st.t.interval(alpha=0.95, df=len(invest_list)-1, loc=np.mean(invest_list), scale=st.sem(invest_list)))
