import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.svm import SVR
from sklearn import metrics
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import pickle
import logging
logging.basicConfig(filename="credit_log.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def train_data():
    try:
        default = pd.read_csv('creditCardFraud.csv')
        default.rename(columns=lambda x: x.lower(), inplace=True)
        logging.info("data imported")
    except Exception as e:
        logging.error(e)
    try:
        # Base values: female, other_education, not_married
        default['grad_school'] = (default['education'] == 1).astype('int')
        default['university'] = (default['education'] == 2).astype('int')
        default['high_school'] = (default['education'] == 3).astype('int')
        default.drop('education', axis=1, inplace=True)

        default['male'] = (default['sex'] == 1).astype('int')
        default.drop('sex', axis=1, inplace=True)

        default['married'] = (default['marriage'] == 1).astype('int')
        default.drop('marriage', axis=1, inplace=True)
        logging.info("base valuses are changed")
    except Exception as e:
        logging.error(e)
    try:
        # For pay features if the <= 0 then it means it was not delayed
        pay_features = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
        for p in pay_features:
            default.loc[default[p] <= 0, p] = 0

        default.rename(columns={'default payment next month': 'default'}, inplace=True)
        logging.info("pay feature updated")
    except Exception as e:
        logging.error(e)
    try:
        x = default.drop('default', axis=1)
        y = default["default"]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        X_train.mean()
        logging.info("the dataset is dividind into a training and testing")
        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(X_train, y_train)

        prediction = linear_regression.predict(X_test)
        lr1 = metrics.mean_absolute_error(y_test, prediction)
        lr2 = metrics.mean_squared_error(y_test, prediction)
        lr3 = np.sqrt(metrics.mean_squared_error(y_test, prediction))
        logging.info(f"the Mean Absolute Error: {lr1}, Mean Squared Error: {lr2}, Root Mean Squared Error: {lr3}")
        supportvector_regressor = SVR()
        supportvector_regressor.fit(X_train, y_train)

        prediction = supportvector_regressor.predict(X_test)
        svr1 = metrics.mean_absolute_error(y_test, prediction)
        svr2 = metrics.mean_squared_error(y_test, prediction)
        svr3 = np.sqrt(metrics.mean_squared_error(y_test, prediction))
        logging.info(f"the Mean Absolute Error: {svr1}, Mean Squared Error: {svr2}, Root Mean Squared Error: {svr3}")
        decision_tree = DecisionTreeRegressor()
        decision_tree.fit(X_train, y_train)

        prediction = decision_tree.predict(X_test)
        dt1 = metrics.mean_absolute_error(y_test, prediction)
        dt2 = metrics.mean_squared_error(y_test, prediction)
        dt3 = np.sqrt(metrics.mean_squared_error(y_test, prediction))
        logging.info(f"the Mean Absolute Error: {dt1}, Mean Squared Error: {dt2}, Root Mean Squared Error: {dt3}")
        n_estimators = 200
        max_depth = 25
        min_samples_split = 15
        min_samples_leaf = 2
        random_forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,min_samples_split=min_samples_split)
        random_forest.fit(X_train, y_train)
        prediction = random_forest.predict(X_test)
        rf1 = metrics.mean_absolute_error(y_test, prediction)
        rf2 = metrics.mean_squared_error(y_test, prediction)
        rf3 = np.sqrt(metrics.mean_squared_error(y_test, prediction))
        logging.info(f"the Mean Absolute Error: {rf1}, Mean Squared Error: {rf2}, Root Mean Squared Error: {rf3}")
        logistic_regression = LogisticRegression(n_jobs=4, random_state=15)
        logistic_regression.fit(X_train, y_train)

        prediction = logistic_regression.predict(X_test)
        log1 = metrics.mean_absolute_error(y_test, prediction)
        log2 = metrics.mean_squared_error(y_test, prediction)
        log3 = np.sqrt(metrics.mean_squared_error(y_test, prediction))
        logging.info(f"the Mean Absolute Error: {log1}, Mean Squared Error: {log2}, Root Mean Squared Error: {log3}")
        with open("logistic_regression", "wb") as f:
            pickle.dump(logistic_regression, f)
        logging.info("the ML model is saved")
    except Exception as e:
        logging.error(e)

def predict_default(limit_bal,age,pay_0,pay_2,pay_3,pay_4,pay_5,pay_6,bill_amt1,bill_amt2,bill_amt3,bill_amt4,bill_amt5,bill_amt6,pay_amt1,pay_amt2,pay_amt3,pay_amt4,pay_amt5,pay_amt6,grad_school,university,high_school,male,married):
    try:
        with open("logistic_regression", "rb") as f:
            lr = pickle.load(f)
        data = {"limit_bal": limit_bal, "age": age, "pay_0": pay_0, "pay_2": pay_2, "pay_3": pay_3, "pay_4": pay_4,
            "pay_5": pay_5, "pay_6": pay_6, "bill_amt1": bill_amt1, "bill_amt2": bill_amt2, "bill_amt3": bill_amt3,
            "bill_amt4": bill_amt4, "bill_amt5": bill_amt5, "bill_amt6": bill_amt6, "pay_amt1": pay_amt1,
            "pay_amt2": pay_amt2, "pay_amt3": pay_amt3, "pay_amt4": pay_amt4, "pay_amt5": pay_amt5,
            "pay_amt6": pay_amt6, "grad_school": grad_school, "university": university, "high_school": high_school,
            "male": male, "married": married}
        client = pd.DataFrame(data, index=[0])
        return lr.predict(client)
    except Exception as e:
        logging.error(e)

