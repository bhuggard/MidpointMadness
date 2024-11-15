import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for kdeplot

# Import your midpoint madness function
def midpoint_madness(qb_name, qb_simulations, over_qbs=[], under_qbs=[]):
    """
    Function to simulate over/under passing yard predictions in Midpoint Madness game.
    """
    results = {}
    combined_prob = 1  # To calculate the combined probability

    # Simulations for the chosen QB
    main_qb_simulations = qb_simulations[qb_name]
    
    # Calculate probabilities for 'over' comparisons
    for over_qb in over_qbs:
        over_qb_simulations = qb_simulations[over_qb]
        prob_main_qb_more = np.mean(np.array(main_qb_simulations) > np.array(over_qb_simulations))
        results[f"{qb_name} > {over_qb}"] = {
            'probability': prob_main_qb_more,
            'american_odds': implied_odds_to_american(prob_main_qb_more)
        }
        # Update the combined probability
        combined_prob *= prob_main_qb_more
    
    # Calculate probabilities for 'under' comparisons
    for under_qb in under_qbs:
        under_qb_simulations = qb_simulations[under_qb]
        prob_main_qb_less = np.mean(np.array(main_qb_simulations) < np.array(under_qb_simulations))
        results[f"{qb_name} < {under_qb}"] = {
            'probability': prob_main_qb_less,
            'american_odds': implied_odds_to_american(prob_main_qb_less)
        }
        # Update the combined probability
        combined_prob *= prob_main_qb_less
    
    # Add combined probability to the results
    results['combined_probability'] = combined_prob
    results['combined_american_odds'] = implied_odds_to_american(combined_prob, cap_large=True)
    
    return results

def implied_odds_to_american(probability, cap_large=True, cap_value=5000):
    """
    Converts probability to American odds format.
    """
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1.")
    
    if probability >= 0.5:
        odds = round(-100 * (probability / (1 - probability)))
    else:
        odds = round(100 * ((1 - probability) / probability))
    
    # Cap large odds if requested
    if cap_large and abs(odds) > cap_value:
        odds = cap_value if odds > 0 else -cap_value
    
    return odds

# Connect to SQLite database
DATABASE_FILE = "midpoint_madness.db"
conn = sqlite3.connect(DATABASE_FILE)

# Load QB data
query = "SELECT * FROM predictions"  # Replace 'predictions' with your table name
df = pd.read_sql(query, conn)

# Convert the 'simulations' column back to a usable format
df['simulations'] = df['simulations'].apply(eval)  # Convert strings back to lists

# Extract the list of QB names globally so it can be used in all sections
qb_names = df['qb_name'].unique()

# Sidebar Navigation
st.sidebar.title("Midpoint Madness")
page = st.sidebar.radio("Navigate to:", ["Welcome", "Midpoint Madness", "Custom Passing Yard Props"])

# Welcome Page
if page == "Welcome":
    st.title("Welcome to Midpoint Madness!")
    st.write("""
        Welcome to **Midpoint Madness**, the game where you test your football knowledge and predictive skills!
        Created by Brandon Huggard
             
        **Midpoint Madness Rules:**
        - Choose a quarterback whose performance you want to predict.
        - Pick an equal number of quarterbacks who will throw for more and fewer passing yards than your selected QB.
        - See if you've created a "Midpoint Madness" scenario and check the probabilities and odds!
        
        **Custom Passing Yard Props:**
        - On the custom props page, you can create your own passing yard line and see the odds based on the simulator.
        - Pick a QB, and guess how many yards they will throw for without going over.
    """)

# Game Page
elif page == "Midpoint Madness":
    st.title("Play Midpoint Madness!")
    
    # Select a QB
    selected_qb = st.selectbox("Select Your Quarterback", qb_names)

    # Display the selected QB's stats
    st.subheader(f"Simulated Passing Yard Distribution for {selected_qb}")
    selected_qb_simulations = df[df['qb_name'] == selected_qb]['simulations'].iloc[0]

    # Create and display the KDE plot using seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(selected_qb_simulations, fill=True, ax=ax)
    ax.set_title(f"How the Simulator Sees Passing Yards for {selected_qb} this Week!", fontsize=14)
    ax.set_xlabel("Passing Yards", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    st.pyplot(fig)

    # Select Over/Under QBs
    st.write("Select an EQUAL number of QBs you think will pass for less and for more yards than your chosen QB.")
    other_qbs = [qb for qb in qb_names if qb != selected_qb]
    over_qbs = st.multiselect("Choose QBs You Think Will Have FEWER Yards", other_qbs)
    filtered_under_qbs = [qb for qb in other_qbs if qb not in over_qbs]
    under_qbs = st.multiselect("Choose QBs You Think Will Have MORE Yards", filtered_under_qbs)

    # Run the game
    if st.button("Play Midpoint Madness"):
        # Ensure the user has selected an equal number of QBs for "more" and "less"
        if len(over_qbs) != len(under_qbs):
            st.error("You must select an equal number of QBs for 'More Passing Yards' and 'Less Passing Yards'.")
        else:
            # Create a dictionary of all simulations for the function
            qb_simulations_dict = dict(zip(df['qb_name'], df['simulations']))

            # Run the midpoint madness function
            results = midpoint_madness(selected_qb, qb_simulations_dict, over_qbs=over_qbs, under_qbs=under_qbs)

            # Display results
            st.markdown("### Results")
            for key, value in results.items():
                if key == 'combined_probability':
                    st.markdown(f"**Likelihood You've Created Midpoint Madness:** {value * 100:.2f}%")
                elif key == 'combined_american_odds':
                    sign = "+" if value > 0 else ""
                    st.markdown(f"**The Price I'd Give You:** {sign}{value:.0f}")
                else:
                    if isinstance(value, dict) and 'american_odds' in value and 'probability' in value:
                        american_odds = value['american_odds']
                        probability = value['probability']
                        sign = "+" if isinstance(american_odds, (int, float)) and american_odds > 0 else ""
                        st.markdown(
                            f"**{key}**: Probability = {probability * 100:.2f}%, "
                            f"American Odds = {sign}{american_odds}"
                        )

# Custom Passing Yard Props Page
elif page == "Custom Passing Yard Props":
    st.title("Custom Passing Yard Props")
    st.write("Pick a quarterback and set your own passing yard lines!")

    # Select QB
    custom_qb = st.selectbox("Choose a Quarterback", qb_names)

    if custom_qb:  # Ensure a QB is selected before proceeding
        # Get simulations for selected QB
        qb_simulations = df[df['qb_name'] == custom_qb]['simulations'].iloc[0]


        # Input custom yardage value
        min_yards = max(int(min(qb_simulations)), 1)  # Ensure the minimum is at least 1
        max_yards = int(max(qb_simulations))
        default_yards = int(np.mean(qb_simulations))

        custom_line = st.slider(
            "Set your own props! How many yards do you think your QB will throw for?",
            min_value=min_yards,
            max_value=max_yards,
            value=default_yards
        )

        # Calculate probability for "More Than" yardage
        prob_over = np.mean(np.array(qb_simulations) > custom_line)
        prob_over = max(0.0001, min(prob_over, 0.9999))  # Avoid 0 or 1 probabilities

        # Convert to American odds
        odds_over = implied_odds_to_american(prob_over, cap_large=True, cap_value=4500)

        # Display results
        st.markdown(f"### Odds {custom_qb} throws for more than {custom_line} yards:")
        st.markdown(f"**Probability:** {prob_over * 100:.2f}%")
        st.markdown(f"**The Price I'd Give You:** {'+' if odds_over > 0 else ''}{odds_over}")
    else:
        st.error("Please select a quarterback to proceed.")
# Close database connection
conn.close()