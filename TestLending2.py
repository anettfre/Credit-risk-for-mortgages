import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

#Helper function to map the values of repaid
def mapping(x):
    if x == 2:
        x = 0
    else:
        x = 1
    return x

## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'

df = pandas.read_csv('../../data/credit/german.data', sep=' ',
                     names=features+[target])
df['repaid'] = df['repaid'].map(mapping)

#df = pandas.read_csv('../../data/credit/german.data', sep=' ', names=features+[target])
#df = pandas.read_csv('../../data/credit/D_valid.csv', sep=' ', names=features+[target])

numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
quantitative_features_2 = []
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
for i in X.columns:
    if i not in numerical_features and i != 'repaid':
        quantitative_features_2.append(i)

encoded_features = list(filter(lambda x: x != target, X.columns))

def qua_noise(X):
    w = np.random.choice([0, 1], size=(len(X), len(quantitative_features_2)), p=[0.7, 0.3])
    X[quantitative_features_2] = (X[quantitative_features_2] + w) % 2
    return X

#Create noise using differential privacy through laplace
#We implement a coin-toss to randomize what data becomes noisy.
def laplace_func(X):
    X_noise = X.copy()
    epsilon = 0.1
    n = np.shape(X)[1]
    for i in numerical_features:
        if np.random.random() > 0.5:
            M = (X[i].max()-X[i].min())
            l = (M*epsilon)/n
            w = np.random.laplace(0, l)    
            X_noise[i] += w

    return X_noise


def add_noise(X_train, X_test):
    X_train_noise = laplace_func(X_train)
    X_test_noise = laplace_func(X_test)
    X_train_noise = qua_noise(X_train_noise)
    X_test_noise = qua_noise(X_test_noise)
    return X_train_noise, X_test_noise


def foreign(data):
    """
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
    print("percentage: ", paid_back/counter)
    print("total: ", 700/1000)


#foreign(df)


def women(data):
    """se hvor mange kvinner og men funksjonen vår faktisk gir lån til
    A92: Female divorced/separated/married
    A93: Male married/widowed
    A94: Male Single
    A95: female single
    'marital status_AXX'
    """
    counter = 0
    paid_back = 0
    for i in range(data.shape[0]):
        if data.iloc[i]['marital status'] == 'A92' or data.iloc[i]['marital status'] == 'A95':
            counter += 1
            if data.iloc[i]['repaid'] == 1:
                paid_back += 1
    print("Number of women in dataset: ", counter)
    print("Number of those who paid back: ", paid_back)
    print("percentage: ", paid_back/counter)
    print("total: ", 700/1000)


women(df)

def bootstrap(data):
    """bootstrap resamples with replacement
    """
    size = int(len(data))
    train = resample(data, n_samples=size, replace=True)
    test = data.drop(train.index)  
    return train[encoded_features], train[target], test[encoded_features], test[target]


## Test function ##
def test_decision_maker(X_test, y_test, interest_rate, decision_maker, woman_not_loan, woman_loan, man_not_loan, man_loan):
    n_test_examples = len(X_test)
    utility = 0

    ## Example test function - this is only an unbiased test if the data has not been seen in training
    total_amount = 0
    total_utility = 0
    #amount_lim = 10000
    #amount_lim_low = 1000
    decision_maker.set_interest_rate(interest_rate)
    
    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])
        good_loan = y_test.iloc[t] # assume the labels are correct
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]
        # If we don't grant the loan then nothing happens
        #if amount < amount_lim:
        #    action = 0
        """    
        A91: Male divorced/separated, no one
        A92: Female divorced/separated/married
        A93: Male married/widowed
        A94: Male Single
        A95: female single, no one
        """
        if (action==1):
            if (good_loan != 1):
                utility -= amount

                if (X_test['marital status_A92'].iloc[t] == 1):
                    woman_not_loan += 1
                if (X_test['marital status_A93'].iloc[t] == 1 or X_test['marital status_A94'].iloc[t] == 1):
                    man_not_loan += 1
            else:    
                utility += amount*(pow(1 + interest_rate, duration) - 1)
                if (X_test['marital status_A92'].iloc[t] == 1):
                    woman_loan += 1
                if (X_test['marital status_A93'].iloc[t] == 1 or X_test['marital status_A94'].iloc[t] == 1):
                    man_loan += 1

        total_utility += utility
        total_amount += amount

    return utility, total_utility/total_amount, woman_not_loan, woman_loan, man_not_loan, man_loan


## Main code

### Setup model
import anadma_banker  #this is a random banker
import random_banker
decision_maker = anadma_banker.AnadmaBanker()
#decision_maker = random_banker.RandomBanker()

interest_rate = 0.05

### Do a number of preliminary tests by splitting the data in parts
from sklearn.model_selection import train_test_split

new_X, hold_out_set = train_test_split(X, test_size=0.2)

for i in range(1):
    n_tests = 1000
    #alpha = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    alpha = [0.1]
    
    for alp in alpha:
        print("\n For alpha = ", alp, " in Multinomil Naive Bayes")
        utility = 0
        investment_return = 0
        woman_loan = 0
        woman_not_loan = 0
        man_loan = 0
        man_not_loan = 0
        utility_list = []
        invest_list = []
        hold_utility = 0
        hold_invest = 0

        for iter in range(n_tests):
            #X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
            X_train, y_train, X_test, y_test = bootstrap(new_X)
            X_train_noise, X_test_noise = add_noise(X_train, X_test)
        
            decision_maker.set_interest_rate(interest_rate)
            decision_maker.fit(X_train_noise, y_train, alp)

            Ui, Ri, woman_not_loan, woman_loan, man_not_loan, man_loan = test_decision_maker(X_test, y_test, interest_rate, 
                                                                 decision_maker, woman_not_loan, 
                                                                 woman_loan, man_not_loan, man_loan)

            hold_Ui, hold_Ri, _, _, _, _ = test_decision_maker(hold_out_set[encoded_features], hold_out_set[target], 
                                                               interest_rate, decision_maker, woman_not_loan, 
                                                               woman_loan, man_not_loan, man_loan)
            utility += Ui
            investment_return += Ri
            hold_utility += hold_Ui
            hold_invest += hold_Ri
            utility_list.append(Ui)
            invest_list.append(Ri)
         
        if (woman_not_loan + woman_loan != 0):
            print("gave loan to number of woman: ", woman_loan/n_tests)
            print("did not give loan to number of woman: ", woman_not_loan/n_tests)
            print("percentage giving loan to women: ", woman_loan / (woman_loan + woman_not_loan))
        if (man_not_loan + man_loan != 0):
            print("gave loan to number of men: ", man_loan/n_tests)
            print("did not give loan to number of men: ", man_not_loan/n_tests)
            print("percentage giving loan to men: ", man_loan / (man_loan + man_not_loan))
         
        plt.bar([1,2,3,4],[woman_loan/n_tests, woman_not_loan/n_tests, man_loan/n_tests, man_not_loan/n_tests],
                tick_label=["Female granted","Female denied","Male granted","Male denied"], color=["green", "red", "green", "red"])
        plt.ylabel("Number of persons")
        plt.show()

        print("Average utility:", utility / n_tests)
        print("95% confidence interval utility", np.percentile(utility_list, 2.5), np.percentile(utility_list, 97.5))
        plt.hist(utility_list)
        plt.xlabel("average utility for different random train/test split")
        plt.ylabel("number of instanses")
        plt.show()

        print("Average return on investment:", investment_return / n_tests)
        print("95% confidence interval return on investment", np.percentile(invest_list, 2.5), np.percentile(invest_list, 97.5))

        print("\n Average utility and return on investment from hold out set: ", hold_utility / n_tests, hold_invest / n_tests)
