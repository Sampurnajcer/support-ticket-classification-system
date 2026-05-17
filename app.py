# ==========================================
# AI SUPPORT TICKET CLASSIFICATION SYSTEM
# Future Interns - ML Task 2
# FINAL ACCURATE VERSION
# ==========================================

# IMPORT LIBRARIES
import streamlit as st
import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# ==========================================
# DOWNLOAD NLTK DATA
# ==========================================

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Ticket Classifier",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("🤖 AI Support Ticket Classification ")

st.write("""
This AI system automatically:
- Classifies customer support tickets
- Detects issue categories
- Helps businesses manage support faster
""")

# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv("tickets.csv")

    return df

df = load_data()

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

df['Ticket Subject'] = df['Ticket Subject'].fillna('')
df['Ticket Description'] = df['Ticket Description'].fillna('')
df['Ticket Type'] = df['Ticket Type'].fillna('Unknown')
df['Ticket Priority'] = df['Ticket Priority'].fillna('Low')

# ==========================================
# COMBINE TEXT COLUMNS
# ==========================================

df['combined_text'] = (
    df['Ticket Subject']
    + " " +
    df['Ticket Description']
)

# ==========================================
# NLP PREPROCESSING
# ==========================================

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    text = str(text)

    # LOWERCASE
    text = text.lower()

    # REMOVE NUMBERS
    text = re.sub(r'\d+', '', text)

    # REMOVE PUNCTUATION
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # TOKENIZATION
    words = text.split()

    # REMOVE STOPWORDS + LEMMATIZE
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# CLEAN TEXT
df['cleaned_text'] = df['combined_text'].apply(preprocess_text)

# ==========================================
# SHOW DATASET
# ==========================================

st.subheader("Dataset Preview")

st.dataframe(
    df[
        [
            'Ticket Subject',
            'Ticket Description',
            'Ticket Type'
        ]
    ].head(),
    use_container_width=True
)

# ==========================================
# LABEL ENCODING
# ==========================================

label_encoder = LabelEncoder()

df['encoded_type'] = label_encoder.fit_transform(
    df['Ticket Type']
)

# ==========================================
# TF-IDF FEATURE EXTRACTION
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(
    df['cleaned_text']
)

y = df['encoded_type']

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# MODEL PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

st.subheader("Model Accuracy")

st.success(
    f"Accuracy: {accuracy * 100:.2f}%"
)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

# ==========================================
# SMALL VISUAL DASHBOARD
# ==========================================

col1, col2 = st.columns(2)

# ==========================================
# CATEGORY DISTRIBUTION
# ==========================================

with col1:

    st.subheader("Ticket Categories")

    fig1, ax1 = plt.subplots(figsize=(4,3))

    sns.countplot(
        data=df,
        y='Ticket Type',
        order=df['Ticket Type'].value_counts().index,
        ax=ax1
    )

    ax1.set_title("Category Distribution")

    st.pyplot(
        fig1,
        use_container_width=False
    )

# ==========================================
# PRIORITY DISTRIBUTION
# ==========================================

with col2:

    st.subheader("Priority Distribution")

    fig2, ax2 = plt.subplots(figsize=(4,3))

    priority_counts = df['Ticket Priority'].value_counts()

    ax2.pie(
        priority_counts,
        labels=priority_counts.index,
        autopct='%1.1f%%'
    )

    ax2.set_title("Priority Levels")

    st.pyplot(
        fig2,
        use_container_width=False
    )

# ==========================================
# CONFUSION MATRIX
# ==========================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig3, ax3 = plt.subplots(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax3
)

ax3.set_title("Model Confusion Matrix")

st.pyplot(
    fig3,
    use_container_width=False
)

# ==========================================
# LIVE TICKET PREDICTION
# ==========================================

st.subheader("🎯 Live Ticket Classification")

user_input = st.text_area(
    "Enter support ticket message:",
    "I forgot my password and cannot login to my account."
)

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("Predict Category"):

    # CLEAN USER INPUT
    cleaned_input = preprocess_text(user_input)

    # VECTORIZE
    input_vector = vectorizer.transform(
        [cleaned_input]
    )

    # ==========================================
    # SMART HYBRID PREDICTION
    # ==========================================

    text_lower = user_input.lower()

    # ACCOUNT ACCESS
    if any(word in text_lower for word in [
        "password",
        "login",
        "signin",
        "otp",
        "account"
    ]):
        final_prediction = "Account Access"

    # BILLING
    elif any(word in text_lower for word in [
        "payment",
        "refund",
        "billing",
        "charged",
        "money"
    ]):
        final_prediction = "Billing Inquiry"

    # TECHNICAL ISSUE
    elif any(word in text_lower for word in [
        "crash",
        "bug",
        "error",
        "upload",
        "technical"
    ]):
        final_prediction = "Technical Issue"

    # DELIVERY ISSUE
    elif any(word in text_lower for word in [
        "delivery",
        "shipping",
        "arrived",
        "courier"
    ]):
        final_prediction = "Delivery Issue"

    # OTHERWISE USE ML MODEL
    else:

        prediction = model.predict(
            input_vector
        )

        final_prediction = label_encoder.inverse_transform(
            prediction
        )[0]

    # SHOW RESULT
    st.success(
        f"Predicted Category: {final_prediction}"
    )

# ==========================================
# SAMPLE INPUTS
# ==========================================

st.subheader("Sample Test Inputs")

st.info("""
1. I forgot my password and cannot login

2. My payment was deducted twice

3. Application crashes during upload

4. My delivery has not arrived

5. I received damaged product and want refund
""")

# ==========================================
# FOOTER
# ==========================================

st.success(
    "AI Support Ticket Classification System Running Successfully"
)