from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_curve
from sklearn.metrics import confusion_matrix

def get_data(file_path):
    df = pd.read_json(file_path, convert_dates=['approx_payout_date','event_created',\
                                               'event_end','event_published',\
                                               'event_start','sale_duration',\
                                               'sale_duration2','user_created'])
    df['event_published'] = df['event_published'].apply(lambda x: (pd.to_datetime(datetime.datetime.fromtimestamp(x).\
                                                    strftime('%Y-%m-%d %H:%M:%S'))) if pd.notnull(x) else x)
    df_new = df.loc[df.acct_type.isin(['premium','fraudster_event','fraudster','fraudster_att']), :].copy()
    df_new['fraud'] = 1
    df_new.loc[df_new.acct_type == 'premium', 'fraud'] = 0
    return df_new

def get_features(df_new):
    df_new['previous_payouts_total'] = df_new.previous_payouts.apply(len)
    df_new['duration'] = df_new.event_created - df_new.user_created
    df_new['duration_days'] = df_new.duration.apply(lambda x: x.days)
    df_new_final = df_new[['duration_days', 'delivery_method', 'num_order', 'num_payouts', 'org_facebook', 'org_twitter', 'sale_duration', \
                       'previous_payouts_total', 'fraud']].dropna()
    return df_new_final


def split_data(df_new_final):
    feature_cols = ['delivery_method', 'num_payouts', 'org_facebook', 'org_twitter', 'sale_duration', \
                'previous_payouts_total']
    X_new_final = df_new_final[feature_cols]
    y_new_final = df_new_final.fraud
    X_train, X_test, y_train, y_test = train_test_split(X_new_final, y_new_final, test_size=0.1)
    return X_train, X_test, y_train, y_test

def model_fit(model, X_train, y_train):
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    model.fit(X_train, y_train)
    return cv, model

def confusion_mt(model, X_test, y_test):
    return confusion_matrix(y_test, model.predict(X_test))

def score(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:,1]
    Accuracy_Score = model.score(X_test, y_test)
    Auc_Score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    Recall_Score = recall_score(y_test, model.predict(X_test))
    Precision_Score = precision_score(y_test, model.predict(X_test))
    return y_pred_proba, Accuracy_Score, Auc_Score, Recall_Score, Precision_Score

def roc_plot(y_test, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('Roc Curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.show()



if __name__ == "__main__":
    file_path = '../data/data.json'
    df_new = get_data(file_path)
    df_new_final = get_features(df_new)
    X_train, X_test, y_train, y_test = split_data(df_new_final)
    model = RandomForestClassifier(n_estimators=200)
    cv, model= model_fit(model, X_train, y_train)
    matrix = confusion_mt(model, X_test, y_test)
    print 'confusion matrix: \n', matrix
    y_pred_proba, Accuracy_Score, Auc_Score, Recall_Score, Precision_Score = score(model, X_test, y_test)
    print 'Accuracy_Score = {:.2f}\nRecall_Score = {:.2f}\nPrecision_Score = {:.2f}\nAuc_Score = {:.4f}'.format(\
        Accuracy_Score,\
        Recall_Score,\
        Precision_Score,\
        Auc_Score)
    print roc_plot(y_test, y_pred_proba)
