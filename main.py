import streamlit as st 
import pandas as pd 
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
st.markdown("### LinkedIn - Who's a User? - Final Project")
s = pd.read_csv("social_media_usage.csv")


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

income = st.selectbox("What is your Annual Salary?",
options = ["Less than $10,000","10 to under $20,000","20 to under $30,000","30 to under $40,000","40 to under $50,000",
           "50 to under $75,000","75 to under $100,000","100 to under $150,000","$150,000 or more?"])


educ2 = st.selectbox("What is your Education level?",
options = ["Less Than High School","Some High School","High School Diploma","Some College (No Degree)", 
           "Associate Degree","College Degree","Some Graduate (No Degree)","Graduate Degree"])

par = st.selectbox("Are you a parent?",
                   options = ["Yes","No"])

marital = st.selectbox("What is your marital status?",
                       options = ["Married","Single"])

gender = st.selectbox("What is your gender?",
                       options = ["Male","Female"])

age = st.slider(label="Enter Age",min_value=1,max_value=97,value=7)


gender_map = {"Male": 1, "Female": 0}
marital_map = {"Single": 0, "Married": 1}
parent_map = {"No": 0, "Yes": 1}
education_map = {
    "Less Than High School": 1,
    "Some High School": 2,
    "High School Diploma": 3,
    "Some College (No Degree)": 4,
    "Associate Degree": 5,
    "College Degree": 6,
    "Some Graduate (No Degree)": 7,
    "Graduate Degree": 8,
}
income_map = {
    "Less than $10,000": 1,
    "10 to under $20,000": 2,
    "20 to under $30,000": 3,
    "30 to under $40,000": 4,
    "40 to under $50,000": 5,
    "50 to under $75,000": 6,
    "75 to under $100,000": 7,
    "100 to under $150,000": 8,
    "$150,000 or more?": 9
}

# Streamlit App
st.title("LinkedIn User Prediction App")

features = [
    age,
    gender_map[gender],
    income_map[income],
    marital_map[marital],
    parent_map[par],
    education_map[educ2],
]

if st.button("Predict"):
    probability = logistic_model.predict_proba([features])[0, 1] 
    is_user = "Yes" if probability > 0.5 else "No"

    st.write(f"**Classification**: {is_user}")
    st.write(f"**Probability**: {probability:.2%}")
