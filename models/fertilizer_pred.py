import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

data = pd.read_csv('./fertilizer.csv')
data.info()

# ======================
# Preprocessing Pipeline
# ======================

y = data['Fertilizer Name'].copy()
X = data.drop('Fertilizer Name', axis=1).copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('nominal', nominal_transformer, ['Soil Type', 'Crop Type'])
], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# ======================
# Training
# ======================

model.fit(X_train, y_train)
joblib.dump(model, 'fertilizer_pred.pkl')
print("Test Accuracy: {:.2f}%".format(model.score(X_test, y_test) * 100))
y_pred = model.predict(X_test)

clr = classification_report(y_test, y_pred)
print("Classification Report:\n----------------------\n", clr)