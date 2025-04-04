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
    st.session_state.username = None
    st.session_state.week = 1
    st.session_state.coins = 100.0
    st.session_state.history = []
    st.session_state.round_complete = False
    st.session_state.wager = 0.0
    st.session_state.streak = 0
    st.session_state.result = None
    st.session_state.selected_qb = None
    st.session_state.over_qbs = []
    st.session_state.under_qbs = []

if 'username' not in st.session_state:
    reset_game()

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
    return f"+{odds}" if odds > 0 else str(odds)

def midpoint_madness_round(main_qb, week, sim_df, over_qbs=[], under_qbs=[]):
    results = []
    correct = 0
    incorrect = 0
    week_df = sim_df[sim_df['week'] == week]
    main_row = week_df[week_df['qb_name'] == main_qb]
    if main_row.empty:
        raise ValueError(f"No simulation data found for QB '{main_qb}' in week {week}")
    main_sim = np.array(main_row.iloc[0]['simulations'])

    parlay_prob = 1
    parlay_american_odds = None

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
        parlay_prob *= prob
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
        parlay_prob *= prob
        correct += int(result)
        incorrect += int(not result)

    if parlay_prob < 1:
        parlay_american_odds = implied_odds_to_american(parlay_prob)

    total_bets = len(results)
    correct_ratio = correct / total_bets if total_bets else 0
    streak_bonus = 1

    if incorrect == 0:
        st.session_state.streak += 1
        if st.session_state.streak == 2:
            streak_bonus = 1
        elif st.session_state.streak >= 3:
            streak_bonus = st.session_state.streak
    else:
        st.session_state.streak = 0

    if incorrect == 0:
        if parlay_prob > 0:
            decimal_odds = 1 / parlay_prob
            total_return = st.session_state.wager * decimal_odds * streak_bonus
            coins_earned = round(total_return, 2)
        else:
            coins_earned = 0.0
    else:
        coins_earned = 0.0

    return {
        'correct': correct,
        'incorrect': incorrect,
        'coins_earned': coins_earned,
        'results': results,
        'streak_bonus': streak_bonus,
        'parlay_prob': parlay_prob if parlay_prob < 1 else None,
        'parlay_odds': parlay_american_odds,
        'winnings_only': round(coins_earned - st.session_state.wager, 2) if incorrect == 0 else 0.0
    }

# ------------------------------
# Streamlit UI (Game Display)
# ------------------------------
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
    st.code(f"I scored {st.session_state.coins:.2f} coins and cashed out at Week {st.session_state.week} in Midpoint Madness! My best move: {best_move if best_move else 'N/A'}")

    if st.button("🔁 Start a New Game"):
        reset_game()
        st.rerun()

    st.stop()

with st.expander("🔁 Reset Game"):
    if st.button("Yes, I'm sure — Reset Game NOW", type="primary"):
        reset_game()
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
            st.session_state.result = midpoint_madness_round(
                st.session_state.selected_qb,
                st.session_state.week,
                predictions_df,
                st.session_state.over_qbs,
                st.session_state.under_qbs
            )
            st.session_state.coins = round(st.session_state.coins + st.session_state.result['coins_earned'] - st.session_state.wager, 2)
            st.session_state.round_complete = True

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