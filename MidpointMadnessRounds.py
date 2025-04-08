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
    return sqlite3.connect("midpoint_madness_rolling.db", check_same_thread=False)

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
def reset_game():
    st.session_state.clear()
    st.session_state.update({
        'username': None,
        'week': 1,
        'coins': 100.0,
        'history': [],
        'round_complete': False,
        'wager': 0.0,
        'streak': 0,
        'result': None,
        'selected_qb': None,
        'over_qbs': [],
        'under_qbs': []
    })
    st.session_state['game_reset'] = True

if 'username' not in st.session_state:
    reset_game()

if st.session_state.get('game_reset'):
    st.session_state['game_reset'] = False
    st.rerun()

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("🏈 Midpoint Madness")
page = st.sidebar.radio("Navigate to:", ["Welcome", "Play Game", "Methodology", "Known Bugs"])

# ------------------------------
# Welcome Page
# ------------------------------
if page == "Welcome":
    st.title("👋 Welcome to Midpoint Madness!")
    st.markdown("""
    Midpoint Madness is a strategic, simulation-driven football prediction game.

    🧠 **How to Play**:
    - Each week, choose a quarterback.
    - Select other QBs who you believe will throw for **more** and **less** yards.
    - You must wager your coins on your picks — score coins by correctly creating "Midpoint Madness" without making any mistakes for a week.
    - After one play per week 

    💰 **Starting Balance**: 100 coins
    🎯 **Goal**: Finish the season with as many coins as possible
    📈 **Scoring**: Based on parlay odds of your correct predictions
    **

    Ready? Head to the **Play Game** tab!
    """)

# ------------------------------
# Methodology Page
# ------------------------------
elif page == "Methodology":
    st.title("🔬 Methodology")
    st.markdown("""
    **How Does the Simulation Work?**

    The core of Midpoint Madness relies on simulated passing yard outcomes generated through a predictive model trained on historical NFL data.

    🏗️ **Model Features**:
    - QB performance metrics (e.g., completion %, air yards)
    - Opponent defense statistics (e.g., EPA allowed, pressure rates)
    - Game context: implied team totals, home/away, weather flags

    📊 **Simulation Details**:
    - Each quarterback has a distribution of 1,000 simulated outcomes per week
    - These simulations are used to determine the probability that one QB will throw for more or less than another
    - Odds are derived using implied probability formulas and adjusted for realism

    💡 **Parlay Logic**:
    - Choosing multiple QBs creates a simulated "parlay" of outcomes
    - The more accurate your picks, the greater your payout — but also the higher the risk!
    
    Have questions? Reach out on GitHub or drop a suggestion in the app!
    """)

# ------------------------------
# Game Page
# ------------------------------
elif page == "Play Game":
    # The entire existing game logic goes here...
    st.title("🏈 Midpoint Madness")

    st.markdown(f"### Week {st.session_state.week} | Coins: {st.session_state.coins:.2f} | Streak: {st.session_state.streak}")

    if st.button("💰 Cash Out and End Game"):
        best_move = None
        best_odds = 100000
        for round_result in st.session_state.history:
            for r in round_result.get('results', []):
                if r['result'] == 'correct' and r['american_odds'] not in ['N/A', None]:
                    try:
                        odds_val = int(r['american_odds'].replace('+', '')) if '+' in r['american_odds'] else abs(int(r['american_odds']))
                        if odds_val < best_odds:
                            best_odds = odds_val
                            best_move = r['comparison'] + f" (Odds: {r['american_odds']})"
                    except:
                        continue

        st.success(f"You cashed out with {st.session_state.coins:.2f} coins at Week {st.session_state.week}! 🎉")
        if best_move:
            st.info(f"🏅 Best Move: {best_move}")
        st.markdown("---")
        st.markdown("Want to share your score? Try copying and pasting:")
        st.code(f"I scored {st.session_state.coins:.2f} coins and made it to Week {st.session_state.week} in Midpoint Madness! My best move: {best_move if best_move else 'N/A'}")

        with st.expander("🔁 Reset Game"):
            if st.button("Yes, I'm sure — Reset Game NOW", type="primary", key="cashout_reset"):
                reset_game()
                st.session_state['game_reset'] = True
                st.rerun()

        st.stop()

    with st.expander("🔁 Reset Game"):
        if st.button("Yes, I'm sure — Reset Game NOW", type="primary"):
            reset_game()
            st.session_state['game_reset'] = True
            st.rerun()

    current_week_df = predictions_df[predictions_df['week'] == st.session_state.week]
    week_qbs = current_week_df['qb_name'].unique()

    st.session_state.selected_qb = st.selectbox("Choose your main QB", week_qbs)
    others = [qb for qb in week_qbs if qb != st.session_state.selected_qb]
    st.session_state.over_qbs = st.multiselect("QBs who will throw for LESS", others)
    st.session_state.under_qbs = st.multiselect("QBs who will throw for MORE", [qb for qb in others if qb not in st.session_state.over_qbs])

    wager_input = st.text_input("Wager this week (e.g., 5.5)", value="10.0")
    try:
        wager_float = round(float(wager_input), 2)
        if wager_float <= 0 or wager_float > st.session_state.coins:
            st.warning(f"Please enter a wager between 0.01 and {st.session_state.coins:.2f}.")
        else:
            st.session_state.wager = wager_float
    except ValueError:
        st.warning("Please enter a valid numeric wager.")

    if not st.session_state.round_complete and st.session_state.wager > 0:
        if st.button("🎯 Lock in Picks"):
            if len(st.session_state.over_qbs) != len(st.session_state.under_qbs):
                st.error("You must select an equal number of over and under QBs.")
            else:
                result = midpoint_madness_round(
                    st.session_state.selected_qb,
                    st.session_state.week,
                    predictions_df,
                    st.session_state.over_qbs,
                    st.session_state.under_qbs
                )

                st.session_state.result = result
                st.session_state.coins = round(st.session_state.coins + result['coins_earned'] - st.session_state.wager, 2)
                st.session_state.round_complete = True
                st.session_state.history.append({
                    'week': st.session_state.week,
                    'main_qb': st.session_state.selected_qb,
                    'over_qbs': st.session_state.over_qbs,
                    'under_qbs': st.session_state.under_qbs,
                    'results': result['results']
                })

    if st.session_state.result:
        result = st.session_state.result
        st.subheader("Results")
        for r in result['results']:
            st.markdown(
                f"**{r['comparison']}**: {r['result']} (Prob: {r['probability']*100:.2f}%, Odds: {r['american_odds']})"
            )

        if result['parlay_prob'] is not None:
            st.markdown(
                f"**Parlay Odds (All picks correct)**: {result['parlay_prob']*100:.2f}% chance, Odds: {result['parlay_odds']}"
            )

        if result['incorrect'] == 0:
            st.success(f"You earned {result['winnings_only']:.2f} coins from your {st.session_state.wager:.2f} coin wager (Total return: {result['coins_earned']:.2f})")
        else:
            st.warning("You did not win this week. Better luck next time!")

        st.info(f"Total Coins: {st.session_state.coins:.2f}")

        if st.button("Next Week"):
            st.session_state.week += 1
            st.session_state.round_complete = False
            st.session_state.result = None

#TO ADD: Leaderboard? Superbowl week logic. Welcome page. description of model/game page. Next add focus has to be welcome page.