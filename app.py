import streamlit as st

# Title and Introduction
st.title("🏈 Midpoint Madness")
st.write("Pick one QB and see how many different players you can stack above and below on passing yards, it's Midpoint Madness!")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Choose Your QB", "Leaderboard", "How to Play"])

# Page Content Based on Sidebar Selection
if page == "Dashboard":
    st.header("🏆 Dashboard")
    st.write("Overview of game details, recent predictions, and general stats.")
    st.write("There's just nothing here right now, there just isn't. IDK but eventually it will be!")
    st.write("Coming soon: Live game predictions and results!")
    
elif page == "Choose Your QB":
    st.header("🎯 Choose Your QB")
    st.write("Select a quarterback and predict if they’ll pass for more or less than the predicted range!")
    
    # Placeholder dropdowns and input controls
    qb = st.selectbox("Select Quarterback", ["QB1", "QB2", "QB3"])
    over_under_line = st.slider("Set Over/Under Line", 100, 500, 250)
    
    # Simulated Curve Display Placeholder
    st.write("Simulated Curve Display: [Placeholder]")
    
    # Action Button
    if st.button("Place Prediction"):
        st.write("Prediction placed! [Placeholder for future interaction]")
    
elif page == "Leaderboard":
    st.header("📈 Leaderboard")
    st.write("Top scores and prediction stats!")
    # Placeholder for leaderboard table or stats
    st.write("Coming soon: Track top players!")
    
elif page == "How to Play":
    st.header("📖 How to Play")
    st.write("Instructions, rules, and tips for predicting NFL QB yardage.")
    # Placeholder for instructions and images/icons