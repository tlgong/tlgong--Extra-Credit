import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from geopy.distance import great_circle
import xgboost as xgb
import datetime

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

y = train['is_fraud']
X = train.drop(['is_fraud'], axis=1)
test_ids = test['id']

X.drop(['id', 'trans_num'], axis=1, inplace=True)
test.drop(['id', 'trans_num'], axis=1, inplace=True)

def create_features(df):
    df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
    df['trans_time'] = pd.to_timedelta(df['trans_time'])
    df['trans_hour'] = df['trans_time'].dt.seconds // 3600
    df['trans_day_of_week'] = df['trans_date'].dt.dayofweek
    df['trans_day'] = df['trans_date'].dt.day
    df['trans_month'] = df['trans_date'].dt.month

    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['age'] = (df['trans_date'] - df['dob']).dt.days // 365

    def calc_distance(row):
        return great_circle((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).kilometers
    df['distance'] = df.apply(calc_distance, axis=1)

    df.drop(['trans_date', 'trans_time', 'dob'], axis=1, inplace=True, errors='ignore')

    return df

X = create_features(X)
test = create_features(test)

X.drop(['unix_time'], axis=1, inplace=True, errors='ignore')
test.drop(['unix_time'], axis=1, inplace=True, errors='ignore')

card_stats = X.groupby('cc_num').agg(
    mean_amt=('amt', 'mean'),
    median_amt=('amt', 'median'),
    count_trans=('amt', 'count'),
    std_amt=('amt', 'std')
).reset_index()

X = X.merge(card_stats, on='cc_num', how='left')
test = test.merge(card_stats, on='cc_num', how='left')

X['std_amt'] = X['std_amt'].fillna(0)
test['std_amt'] = test['std_amt'].fillna(0)

for col in ['merchant','city','street','first','last','job']:
    freq = X[col].value_counts().to_dict()
    X[col + '_freq'] = X[col].map(freq)
    test[col + '_freq'] = test[col].map(freq)

X.drop(['merchant','city','street','first','last','job'], axis=1, inplace=True, errors='ignore')
test.drop(['merchant','city','street','first','last','job'], axis=1, inplace=True, errors='ignore')

X['amt_distance_inter'] = X['amt'] * X['distance']
X['age_amt_inter'] = X['age'] * X['amt']

test['amt_distance_inter'] = test['amt'] * test['distance']
test['age_amt_inter'] = test['age'] * test['amt']

numeric_features = [
    'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
    'trans_hour', 'trans_day_of_week', 'trans_day', 'trans_month', 'age', 'distance',
    'mean_amt','median_amt','count_trans','std_amt',
    'merchant_freq','city_freq','street_freq','first_freq','last_freq','job_freq',
    'amt_distance_inter','age_amt_inter'
]

cat_features = ['category','gender','state']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=16, stratify=y)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, cat_features),
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))
])

param_grid = {
    'classifier__n_estimators': [300,400],
    'classifier__max_depth': [4,6, 10],
    'classifier__learning_rate': [ 0.2,0.25,0.3],
    'classifier__colsample_bytree': [0.7, 1.0],
    'classifier__subsample': [1.0]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_val_pred = best_model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_acc)
print("Best Parameters:", grid_search.best_params_)

test_pred = best_model.predict(test)
submission = sample_submission.copy()
submission['is_fraud'] = test_pred
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
