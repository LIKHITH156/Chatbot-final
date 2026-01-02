
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "question": [
        "What is cricket?",
        "How many players are in a football team?",
        "What is an offside in football?",
        "How long is a cricket match?",
        "What is a hat-trick?",
        "How many sets are in tennis?",
        "What is a free throw in basketball?",
        "What is VAR in football?",
        "What is IPL?",
        "Who is called the god of cricket?"
    ],
    "answer": [
        "Cricket is a bat-and-ball game played between two teams of 11 players.",
        "A football team has 11 players.",
        "Offside is a rule to prevent unfair advantage in football.",
        "A cricket match can last from 3 hours to 5 days.",
        "A hat-trick means three goals or wickets in a row.",
        "A tennis match has 3 or 5 sets.",
        "A free throw is awarded after a foul in basketball.",
        "VAR stands for Video Assistant Referee.",
        "IPL is the Indian Premier League.",
        "Sachin Tendulkar is known as the God of Cricket."
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    if similarity[0][index] > 0.3:
        return df.iloc[index]["answer"]
    else:
        return "Sorry, I don't know the answer to that yet."

st.set_page_config(page_title="Sports FAQ Chatbot")
st.title("🏆 Sports FAQ Chatbot")

user_input = st.text_input("Ask a sports question:")

if st.button("Ask"):
    if user_input:
        st.success(chatbot_response(user_input))
    else:
        st.warning("Please enter a question.")
