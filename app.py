
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
        "Who is called the god of cricket?",
        
        "What is basketball?",
        "How many players are on a basketball court?",
        "What is a touchdown?",
        "What is a penalty kick?",
        "What is a red card?",
        "What is badminton?",
        "How many overs are in T20 cricket?",
        "What is a no-ball?",
        "What is a wicket?",
        "What is a bowler?",
        
        "What is swimming?",
        "What is athletics?",
        "What is a marathon?",
        "What is a relay race?",
        "What is a goalkeeper?",
        "What is table tennis?",
        "What is squash?",
        "What is volleyball?",
        "How many players in volleyball?",
        "What is a spike?",
        
        "What is chess?",
        "What is checkmate?",
        "What is a draw in chess?",
        "What is castling?",
        "What is boxing?",
        "What is a knockout?",
        "What is MMA?",
        "What is judo?",
        "What is karate?",
        "What is wrestling?",
        
        "What is kabaddi?",
        "What is a raid?",
        "What is kho-kho?",
        "What is fencing?",
        "What is archery?",
        "What is shooting?",
        "What is weightlifting?",
        "What is powerlifting?",
        "What is gymnastics?",
        "What is skating?",
        
        "What is cycling?",
        "What is Formula 1?",
        "What is MotoGP?",
        "What is rally racing?",
        "What is esports?",
        "What is PUBG?",
        "What is FIFA?",
        "What is a referee?",
        "What is an umpire?",
        "What is a coach?",
        
        "What is teamwork?",
        "What is sportsmanship?",
        "What is fair play?",
        "What is endurance?",
        "What is stamina?",
        "What is agility?",
        "What is flexibility?",
        "What is strength?",
        "What is fitness?",
        "What is warm-up?",
        
        "What is cool down?",
        "What is hydration?",
        "What is nutrition?",
        "What is protein?",
        "What is carbohydrate?",
        "What is fat?",
        "What is BMI?",
        "What is injury?",
        "What is first aid?",
        "What is recovery?",
        
        "What is yoga?",
        "What is meditation?",
        "What is mental fitness?",
        "What is concentration?",
        "What is focus?",
        "What is reaction time?",
        "What is coordination?",
        "What is balance?",
        "What is speed?",
        "What is power?",
        
        "What is endurance training?",
        "What is strength training?",
        "What is cardio?",
        "What is HIIT?",
        "What is stretching?",
        "What is flexibility training?",
        "What is aerobic exercise?",
        "What is anaerobic exercise?",
        "What is metabolism?",
        "What is calorie?",
        
        "What is diet?",
        "What is healthy food?",
        "What is junk food?",
        "What is hydration level?",
        "What is fatigue?",
        "What is recovery time?",
        "What is sports psychology?",
        "What is motivation?",
        "What is discipline?",
        "What is consistency?"
    ],

    "answer": [
        "Cricket is a bat-and-ball game played between two teams of eleven players.",
        "A football team has 11 players.",
        "Offside is a rule to prevent unfair attacking advantage.",
        "A cricket match duration depends on the format played.",
        "A hat-trick means scoring three goals or taking three wickets consecutively.",
        "Tennis matches are usually played in best of three or five sets.",
        "A free throw is an unopposed shot in basketball after a foul.",
        "VAR is Video Assistant Referee used to review decisions.",
        "IPL is the Indian Premier League, a T20 cricket tournament.",
        "Sachin Tendulkar is called the God of Cricket.",
        
        "Basketball is a team sport played with a ball and hoop.",
        "There are 10 players on a basketball court.",
        "A touchdown is a scoring method in American football.",
        "A penalty kick is taken after a foul inside the box.",
        "A red card means a player is sent off.",
        "Badminton is a racquet sport played with a shuttlecock.",
        "A T20 match has 20 overs per team.",
        "A no-ball is an illegal delivery in cricket.",
        "A wicket refers to the stumps or dismissal of a batsman.",
        "A bowler delivers the ball in cricket.",
        
        "Swimming is a water-based sport.",
        "Athletics includes running, jumping, and throwing events.",
        "A marathon is a long-distance running race.",
        "A relay race involves a team passing a baton.",
        "A goalkeeper defends the goal.",
        "Table tennis is played with paddles and a small ball.",
        "Squash is played in an enclosed court.",
        "Volleyball is a team sport played over a net.",
        "Volleyball has six players per team.",
        "A spike is an attacking shot in volleyball.",
        
        "Chess is a strategy board game.",
        "Checkmate ends the game in chess.",
        "A draw means no winner.",
        "Castling is a special king move in chess.",
        "Boxing is a combat sport using fists.",
        "A knockout ends a boxing match.",
        "MMA is mixed martial arts.",
        "Judo is a Japanese martial art.",
        "Karate is a striking martial art.",
        "Wrestling is a grappling sport.",
        
        "Kabaddi is a contact team sport.",
        "A raid is an attack in kabaddi.",
        "Kho-kho is a traditional Indian sport.",
        "Fencing is a sword-based sport.",
        "Archery is shooting arrows at targets.",
        "Shooting is a precision sport.",
        "Weightlifting involves lifting heavy weights.",
        "Powerlifting focuses on strength lifts.",
        "Gymnastics involves flexibility and balance.",
        "Skating is gliding on ice or wheels.",
        
        "Cycling is riding a bicycle competitively.",
        "Formula 1 is a car racing sport.",
        "MotoGP is motorcycle racing.",
        "Rally racing is off-road racing.",
        "Esports are competitive video games.",
        "PUBG is an online battle game.",
        "FIFA is an international football body.",
        "A referee enforces game rules.",
        "An umpire supervises sports matches.",
        "A coach trains athletes.",
        
        "Teamwork means working together.",
        "Sportsmanship means fair behavior.",
        "Fair play means playing honestly.",
        "Endurance is the ability to last long.",
        "Stamina is physical energy.",
        "Agility is quick movement ability.",
        "Flexibility is joint movement ability.",
        "Strength is muscle power.",
        "Fitness is overall physical health.",
        "Warm-up prepares the body for exercise.",
        
        "Cool down relaxes muscles after exercise.",
        "Hydration means maintaining water levels.",
        "Nutrition is intake of healthy food.",
        "Protein helps muscle growth.",
        "Carbohydrates provide energy.",
        "Fat provides stored energy.",
        "BMI measures body mass index.",
        "Injury is physical harm.",
        "First aid is immediate medical help.",
        "Recovery is rest after activity.",
        
        "Yoga improves physical and mental health.",
        "Meditation improves concentration.",
        "Mental fitness improves focus.",
        "Concentration means attention on task.",
        "Focus means mental clarity.",
        "Reaction time is response speed.",
        "Coordination is body control.",
        "Balance is body stability.",
        "Speed is quick movement.",
        "Power is strength with speed.",
        
        "Endurance training increases stamina.",
        "Strength training builds muscles.",
        "Cardio improves heart health.",
        "HIIT is high-intensity training.",
        "Stretching improves flexibility.",
        "Flexibility training reduces injury.",
        "Aerobic exercise uses oxygen.",
        "Anaerobic exercise is high intensity.",
        "Metabolism is energy conversion.",
        "A calorie measures energy.",
        
        "Diet is daily food intake.",
        "Healthy food improves health.",
        "Junk food lacks nutrition.",
        "Hydration level shows water balance.",
        "Fatigue means tiredness.",
        "Recovery time is rest period.",
        "Sports psychology studies athlete behavior.",
        "Motivation drives performance.",
        "Discipline means self-control.",
        "Consistency means regular effort."
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
