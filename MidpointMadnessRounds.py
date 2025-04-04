import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Database setup
# ------------------------------
@st.cache_resource
def get_connection():
    return sqlite3.connect("midpoint_madness.db", check_same_thread=False)

conn = get_connection()

# Load predictions data
predictions_df = pd.read_sql("SELECT * FROM predictions", conn)
predictions_df['pyoe_list'] = predictions_df['pyoe_list'].apply(json.loads)
predictions_df['simulations'] = predictions_df['simulations'].apply(json.loads)

available_weeks = sorted(predictions_df['week'].unique())
all_qbs = predictions_df['qb_name'].unique()

# ------------------------------
# Session state initialization
# ------------------------------
if 'username' not in st.session_state:
    st.session_state.username = None
if 'week' not in st.session_state:
    st.session_state.week = 1
if 'lives' not in st.session_state:
    st.session_state.lives = 3
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------------------
# Utility Functions
# ------------------------------
def implied_odds_to_american(probability, cap_large=True, cap_value=5000):
    if probability <= 0 or probability >= 1:
        return 'N/A'
    if probability >= 0.5:
        odds = round(-100 * (probability / (1 - probability)))
    else:
        odds = round(100 * ((1 - probability) / probability))
    if cap_large and abs(odds) > cap_value:
        odds = cap_value if odds > 0 else -cap_value
    return odds

def midpoint_madness_round(main_qb, week, sim_df, over_qbs=[], under_qbs=[]):
    results = []
    correct = 0
    incorrect = 0
    week_df = sim_df[sim_df['week'] == week]
    main_row = week_df[week_df['qb_name'] == main_qb]
    if main_row.empty:
        raise ValueError(f"No simulation data found for QB '{main_qb}' in week {week}")
    main_sim = np.array(main_row.iloc[0]['simulations'])

    for over_qb in over_qbs:
        comp_sim = np.array(week_df[week_df['qb_name'] == over_qb].iloc[0]['simulations'])
        prob = np.mean(main_sim > comp_sim)
        result = prob > 0.5
        results.append({
            'comparison': f"{main_qb} > {over_qb}",
            'probability': prob,
            'american_odds': implied_odds_to_american(prob),
            'result': 'correct' if result else 'incorrect'
        })
        correct += int(result)
        incorrect += int(not result)

    for under_qb in under_qbs:
        comp_sim = np.array(week_df[week_df['qb_name'] == under_qb].iloc[0]['simulations'])
        prob = np.mean(main_sim < comp_sim)
        result = prob > 0.5
        results.append({
            'comparison': f"{main_qb} < {under_qb}",
            'probability': prob,
            'american_odds': implied_odds_to_american(prob),
            'result': 'correct' if result else 'incorrect'
        })
        correct += int(result)
        incorrect += int(not result)

    return {
        'correct': correct,
        'incorrect': incorrect,
        'lives_lost': incorrect,
        'points_earned': correct,
        'results': results
    }

# ------------------------------
# Streamlit UI
# ------------------------------
st.sidebar.title("Midpoint Madness")
page = st.sidebar.radio("Navigate to:", ["Welcome", "Play Game"])

if page == "Welcome":
    st.title("Welcome to Midpoint Madness!")
    st.write("""
    Welcome to **Midpoint Madness**, the football survival game!
    - Each week, choose your QB.
    - Pick an equal number of QBs you think will throw for more and less yards.
    - Survive through all 22 weeks with just 3 lives!
    """)

elif page == "Play Game":
    st.title(f"Week {st.session_state.week} — Midpoint Madness")

    if st.session_state.lives <= 0:
        st.error("Game over! You've run out of lives.")
        st.stop()
    if st.session_state.week > max(available_weeks):
        st.success("You've made it through the season!")
        st.stop()

    current_week_df = predictions_df[predictions_df['week'] == st.session_state.week]
    week_qbs = current_week_df['qb_name'].unique()

    selected_qb = st.selectbox("Choose your main QB", week_qbs)

    # Plot their distribution
    selected_sim = current_week_df[current_week_df['qb_name'] == selected_qb]['simulations'].iloc[0]
    fig, ax = plt.subplots()
    sns.kdeplot(selected_sim, fill=True, ax=ax)
    ax.set_title(f"Simulated Yards for {selected_qb} in Week {st.session_state.week}")
    st.pyplot(fig)

    others = [qb for qb in week_qbs if qb != selected_qb]
    over_qbs = st.multiselect("Pick QBs who will throw for LESS", others)
    under_qbs = st.multiselect("Pick QBs who will throw for MORE", [qb for qb in others if qb not in over_qbs])

    if st.button("Lock in Picks!"):
        if len(over_qbs) != len(under_qbs):
            st.error("You must pick an equal number of over and under QBs.")
        else:
            result = midpoint_madness_round(selected_qb, st.session_state.week, predictions_df, over_qbs, under_qbs)
            st.session_state.score += result['points_earned']
            st.session_state.lives -= result['lives_lost']
            st.session_state.history.append({
                'week': st.session_state.week,
                'main_qb': selected_qb,
                'over_qbs': over_qbs,
                'under_qbs': under_qbs,
                'result': result
            })
            st.session_state.week += 1

            # Display round results
            st.subheader("Results")
            for r in result['results']:
                st.markdown(f"**{r['comparison']}**: {r['result']} (Prob: {r['probability']*100:.2f}%, Odds: {r['american_odds']})")

            st.success(f"You earned {result['points_earned']} point(s) and lost {result['lives_lost']} lives.")
            st.info(f"Score: {st.session_state.score} | Lives: {st.session_state.lives}")