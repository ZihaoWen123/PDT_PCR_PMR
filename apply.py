# coding=gbk
import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


def knn_impute(df, n_neighbors=3):

    new_column = [0 for i in range(len(df))]

    df.insert(0, 'label', new_column)
    groups = df.groupby('label')

    imputer = KNNImputer(n_neighbors=n_neighbors)

    imputed_dfs = []
    for label, group in groups:
        for i, key in enumerate(group):
            if i != 0:
                if n_neighbors > 0:
                    try:
                        group[key] = imputer.fit_transform(np.array(group[key]).reshape(-1, 1))
                    except:
                        stop=1
                else:
                    group.loc[:, str(key)] = group.loc[:, str(key)].fillna(group.loc[:, str(key)].median())

        imputed_dfs.append(group)

    df = pd.concat(imputed_dfs, ignore_index=True)
    df = df.drop(df.columns[0], axis=1)

    # df = df.sample(frac=1, random_state=43).reset_index(drop=True)

    return df

def replace_outliers_with_nan(data, threshold=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    data = np.where((data < lower_bound) | (data > upper_bound), np.nan, data)
    return data


if __name__ == '__main__':
    useknn_impute = True
    model_type = 'mo'  # mo, cu 注意是小写Zircon+Composition+DB+-+Fertility.csv
    file_name = 'WS23-6-1.csv'

    loaded_scaler = joblib.load(f'{model_type}_scaler.pkl')
    loaded_model = joblib.load(f'{model_type}_model.pkl')

    df = pd.read_csv(file_name, encoding='gbk')
    df1 = pd.read_csv('PMR.csv', encoding='gbk')

    columns_to_drop = ['La', 'Pr']
    df = df.drop(columns=columns_to_drop)

    if model_type in ['fertility', 'cu']:
        columns_to_drop = ['Nb', 'Ta']
    else:
        columns_to_drop = ['Y', 'Nb', 'Ta', 'Hf/Y', '△FMQ', '(EuN/EuN*)/Y']
    df = df.drop(columns=columns_to_drop)
    columns_difference = set(df.columns) - set(df1.columns)

    for col in df.columns:
        if col != 'label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = replace_outliers_with_nan(df[col])

    df = df.dropna(how='all')
    if useknn_impute:
        # X_train, X_test, y_train, y_test = knn_impute(df, 'label')
        df = knn_impute(df)

    X_new_scaled = loaded_scaler.transform(df.values)

    y_pred = loaded_model.predict(X_new_scaled)

    class_probabilities = loaded_model.predict_proba(X_new_scaled)

    result_df = pd.DataFrame({
        'Predicted_Label': y_pred,
        'Class_Probabilities': [list(probs) for probs in class_probabilities]
    })

    result_df.to_csv(f'{model_type}_apply_res.csv', index=False)