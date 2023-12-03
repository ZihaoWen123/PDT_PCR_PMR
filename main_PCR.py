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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')



def knn_impute(df, label_col, n_neighbors=3, test_size=0.1):

    groups = df.groupby(label_col)

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
    df = df.sample(frac=1, random_state=43).reset_index(drop=True)

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
    usehyparams_search = False
    useknn_impute = True
    file_name = 'PCR_Database.csv'

    df = pd.read_csv(file_name, encoding='gbk')

    columns_to_drop = ['La', 'Pr']
    df = df.drop(columns=columns_to_drop)

    label_encoder = LabelEncoder()
    df['label'] = df['label'].astype(str)
    df['label'] = label_encoder.fit_transform(df['label'])

    for col in df.columns:
        if col != 'label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = replace_outliers_with_nan(df[col])

    nan_percentage = df.isna().mean() * 100
    threshold = 30

    columns_to_drop = nan_percentage[nan_percentage > threshold].index.tolist()

    df = df.drop(columns=columns_to_drop)

    df = df.dropna(how='all')

    if useknn_impute:
        df = knn_impute(df, 'label')
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df['label'], test_size=0.1,
                                                            random_state=42)
        X, y = df.iloc[:, 1:], df['label']
    else:
        X, y = df.iloc[:, 1:], df['label']
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df['label'], test_size=0.1,
                                                            random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.ones(len(X_train))

    for class_label, weight in enumerate(class_weights):
        sample_weights[y_train == class_label] = weight

    if not usehyparams_search:
        clf = XGBClassifier(
            objective='multi:softmax',
            num_class=5,
            max_depth=8,
            n_estimators=1000,
            random_state=42
        )
    else:
        clf = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', num_class=5, use_label_encoder=False,
                              random_state=42)

        param_space = {
            'learning_rate': Real(0.01, 0.5),
            'max_depth': Integer(1, 10),
            'n_estimators': Integer(50, 1200),
            'min_child_weight': Integer(1, 10),
            'subsample': Real(0.1, 1.0),
            'colsample_bytree': Real(0.1, 1.0),
            'gamma': Real(0, 1)
        }

        opt = BayesSearchCV(
            clf,
            param_space,
            n_iter=10,
            cv=5,
            scoring='f1_macro',
            n_jobs=1,
            random_state=42,
            verbose=0
        )

        opt.fit(np.row_stack((X_train, X_test)), np.concatenate((y_train, y_test)))

        search_res = np.column_stack(
            (np.array(opt.optimizer_results_[0]['x_iters']), -1 * opt.optimizer_results_[0]['func_vals']))

        column_names = ['colsample_bytree', 'gamma', "learning_rate", "max_depth", "min_child_weight", "n_estimators",
                        'subsample', "score"]
        search_res_df = pd.DataFrame(search_res, columns=column_names)

        search_res_df.to_csv(f'search_res.csv', index=False)

        clf.set_params(**opt.best_params_)
        print('best-f1:', opt.best_score_)
        print('best-params:', opt.best_params_)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    f1_scorer = make_scorer(f1_score, average='macro')

    scores = cross_val_score(clf, X, y, cv=kf, scoring=f1_scorer)

    for i, score in enumerate(scores):
        print(f"Fold {i+1} F1 Score: {score}")

    print(f"Mean F1 Score: {scores.mean()}")

    clf.fit(X_train, y_train, sample_weights)

    y_pred = clf.predict(X_test)

    feature_importance = clf.feature_importances_

    accuracy = accuracy_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    joblib.dump(scaler, 'cu_scaler.pkl')
    joblib.dump(clf, 'cu_model.pkl')

    df = pd.read_csv(file_name, encoding='gbk')
    columns_to_drop1 = ['La', 'Pr']
    df = df.drop(columns=columns_to_drop1)
    top_10_indices = np.argsort(feature_importance)[::-1][:]

    df = df.drop(columns=columns_to_drop)

    top_10_feature_names = [df.columns[1:][i] for i in top_10_indices]

    top_10_feature_importances = feature_importance[top_10_indices]/np.sum(feature_importance)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_10_feature_names, top_10_feature_importances, color='skyblue')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature Name')
    plt.title('Top 10 Feature Importances in XGBoost Model')
    plt.gca().invert_yaxis()

    for bar, importance_score in zip(bars, top_10_feature_importances):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{importance_score:.3f}', ha='left', va='center')

    plt.savefig('feature importance.png',dpi=600)
    plt.show()

    explainer = shap.Explainer(clf)

    shap_values = explainer.shap_values(X_train)

    feature_importance = np.mean(np.sum(np.abs(shap_values), axis=0), axis=0)
    feature_importance = feature_importance / np.sum(feature_importance)

    feature_importance_df = pd.DataFrame({
        'feature': df.columns[1:],
        'importance': feature_importance
    })

    top10_features = feature_importance_df.sort_values(by='importance', ascending=False)#.head(10)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top10_features['feature'], top10_features['importance'], color='skyblue')
    plt.xlabel('Importance Score')
    plt.title('Top 10 Important Features based on shap')
    plt.gca().invert_yaxis()

    for bar, importance_score in zip(bars, top10_features['importance']):

        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{importance_score:.3f}', ha='left', va='center')

    plt.savefig('feature importance_shap.png', dpi=600)
    plt.show()
    plt.show()

    class_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    print("Class to Index Mapping:")
    for class_name, index in class_mapping.items():
        print(f"{class_name}: {index}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(np.unique(y_test))), class_mapping.keys())
    plt.yticks(np.arange(len(np.unique(y_test))), class_mapping.keys())
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    for i in range(len(np.unique(y_test))):
        for j in range(len(np.unique(y_test))):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.savefig('confusionmatrix.png', dpi=600)
    plt.show()

    class_probabilities = clf.predict_proba(X_test)

    result_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'Class_Probabilities': [list(probs) for probs in class_probabilities]
    })

    result_df.to_csv('probability_Cu.csv', index=False)