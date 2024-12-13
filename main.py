import streamlit as st 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
st.markdown("### LinkedIn Usage")
s = pd.read_csv("social_media_usage.csv")
s.shape


def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x
  
ss = pd.DataFrame()
ss['sm_li'] = s['web1h'].apply(clean_sm)
ss['income'] = s['income'].where(s['income'] <= 9, np.nan)
ss['educ2'] = s['educ2'].where(s['educ2'] <= 8, np.nan)
ss['marital'] = s['marital'].apply(clean_sm)
ss['gender'] = s['gender'].apply(clean_sm)
ss['par'] = s['par'].apply(clean_sm)
ss['age'] = s['age'].apply(lambda x: x if x <= 98 else np.nan)
ss.dropna(inplace=True)


y = ss['sm_li']

X = ss.drop(columns=['sm_li'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logistic_model = LogisticRegression(class_weight='balanced', random_state=42)

logistic_model.fit(X_train, y_train)



y_pred = logistic_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

report = classification_report(y_test, y_pred, target_names=['Not Using LinkedIn', 'Using LinkedIn'], output_dict=True)

print("Classification Report:")
print(report)

precision_sklearn = report['Using LinkedIn']['precision']
recall_sklearn = report['Using LinkedIn']['recall']
f1_score_sklearn = report['Using LinkedIn']['f1-score']


