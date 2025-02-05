#Import dependencies

#NFL pbp data
import nfl_data_py as nfl

#Basics / visualizations
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#Machine learning tools
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Stats Stuff
from scipy import stats
from scipy.stats import t
from scipy.stats import truncnorm

#Turn off max columns for pandas DataFrame
pd.set_option('display.max_columns', None)

#Turn off when showing off, switch comments to change
pd.options.mode.chained_assignment = None  # Disable the warning
# pd.options.mode.chained_assignment = 'warn'



def calculate_implied_totals(df):
    """
    Calculate the implied home and away team totals based on the spread and total lines.
    """
    #Implied totals based on the total and spread lines
    df['implied_home_total'] = (df['total_line'] + df['spread_line']) / 2
    df['implied_away_total'] = (df['total_line'] - df['spread_line']) / 2
    
    return df

def format_passer_name(qb_name):
    if pd.isna(qb_name):  #Check if the name is NaN
        return ""
    
    name_parts = qb_name.split()
    
    #Extract the first name and last name
    first_name = name_parts[0]
    last_name = name_parts[-1]  # Last name should always be the last part
    
    return f"{first_name[0]}.{last_name}"

def calculate_offensive_ewma(passer_df):
    """
    Calculates EWMA for offensive columns using previous weeks' data, ensuring no leakage by excluding the current week.
    Takes into account multiple seasons.
    """
    #Sort by passer, season, and week
    passer_df = passer_df.sort_values(by=['passer_player_name', 'season', 'week'])

    #Calculate the exponentially weighted moving average for each offensive feature, excluding the current week
    passer_df['completion_percentage_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['completion_percentage']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    passer_df['pass_attempts_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['pass_attempts']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    passer_df['air_yards_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['air_yards']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    passer_df['yards_after_catch_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['yards_after_catch']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    passer_df['epa_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['epa']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    passer_df['interception_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['interception']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    passer_df['qb_hit_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['qb_hit']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    passer_df['sack_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['sack']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    passer_df['pass_touchdown_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['pass_touchdown']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    passer_df['passing_yards_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['passing_yards']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    passer_df['cpoe_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['cpoe']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    return passer_df

def pass_cleaner(passer_df):
    """
    Preps passer df for merging; drops unnecessary columns
    """
    passer_df.drop(columns=['home_team', 'away_team', 'complete_pass', 'incomplete_pass', 'completion_percentage', 'air_yards', 'yards_after_catch', 'epa', 
                                    'interception', 'qb_hit', 'sack', 'pass_touchdown', 'cpoe', 'home_team', 'away_team', 
                                    'complete_pass', 'incomplete_pass'], inplace=True)
    
    return passer_df

def calculate_defensive_ewma(defense_df):
    """
    Calculates EWMA for defensive columns using previous weeks' data (excluding the current week).
    """
    #Sort by 'defteam', 'season', and 'week' in ascending order (to ensure time order)
    defense_df = defense_df.sort_values(by=['defteam', 'season', 'week'])

    #Ensure proper grouping by both defteam and season
    defense_df['completion_percentage_ewma'] = defense_df.groupby(['defteam', 'season'])['completion_percentage']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())
    
    defense_df['pass_attempts_ewma'] = defense_df.groupby(['defteam', 'season'])['pass_attempts']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    defense_df['air_yards_ewma'] = defense_df.groupby(['defteam', 'season'])['air_yards']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    defense_df['yards_after_catch_ewma'] = defense_df.groupby(['defteam', 'season'])['yards_after_catch']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    defense_df['epa_ewma'] = defense_df.groupby(['defteam', 'season'])['epa']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    defense_df['interception_ewma'] = defense_df.groupby(['defteam', 'season'])['interception']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    defense_df['qb_hit_ewma'] = defense_df.groupby(['defteam', 'season'])['qb_hit']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    defense_df['sack_ewma'] = defense_df.groupby(['defteam', 'season'])['sack']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    defense_df['pass_touchdown_ewma'] = defense_df.groupby(['defteam', 'season'])['pass_touchdown']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    defense_df['passing_yards_ewma'] = defense_df.groupby(['defteam', 'season'])['passing_yards']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    defense_df['cpoe_ewma'] = defense_df.groupby(['defteam', 'season'])['cpoe']\
        .transform(lambda x: x.shift().ewm(min_periods=1, span=5).mean())

    return defense_df

def def_cleaner(defense_df):
    """
    Preps passer df for merging; drops unnecessary columns
    """
    #Drop the non-ewma columns
    defense_df.drop(columns=['passing_yards','completion_percentage',
                            'air_yards', 'yards_after_catch', 'epa',     
                            'interception', 'qb_hit', 'sack', 'pass_touchdown', 'pass_attempts', 'cpoe', 'complete_pass', 'incomplete_pass',
                            'home_team', 'away_team'
                            ], inplace=True)
    
    return defense_df

def calculate_ewma_tester_off(passer_df):
    """
    Calculates EWMA for offensive columns using the current and previous weeks' data, including the current week.
    Takes into account multiple seasons.
    """
    #Sort by passer, season, and week
    passer_df = passer_df.sort_values(by=['passer_player_name', 'season', 'week'])

    #Calculate the exponentially weighted moving average for each offensive feature, including the current week
    passer_df['completion_percentage_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['completion_percentage']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    passer_df['pass_attempts_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['pass_attempts']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    passer_df['air_yards_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['air_yards']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    passer_df['yards_after_catch_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['yards_after_catch']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    passer_df['epa_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['epa']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    passer_df['interception_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['interception']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    passer_df['qb_hit_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['qb_hit']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    passer_df['sack_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['sack']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    passer_df['pass_touchdown_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['pass_touchdown']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    passer_df['passing_yards_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['passing_yards']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    passer_df['cpoe_ewma'] = passer_df.groupby(['passer_player_name', 'season'])['cpoe']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    return passer_df

def calculate_ewma_tester_def(defense_df):
    """
    Calculates EWMA for defensive columns using previous weeks' data (excluding the current week).
    """
    #Sort by 'defteam', 'season', and 'week' in ascending order (to ensure time order)
    defense_df = defense_df.sort_values(by=['defteam', 'season', 'week'])

    #Ensure proper grouping by both defteam and season
    defense_df['completion_percentage_ewma'] = defense_df.groupby(['defteam', 'season'])['completion_percentage']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())
    
    defense_df['pass_attempts_ewma'] = defense_df.groupby(['defteam', 'season'])['pass_attempts']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    defense_df['air_yards_ewma'] = defense_df.groupby(['defteam', 'season'])['air_yards']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    defense_df['yards_after_catch_ewma'] = defense_df.groupby(['defteam', 'season'])['yards_after_catch']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    defense_df['epa_ewma'] = defense_df.groupby(['defteam', 'season'])['epa']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    defense_df['interception_ewma'] = defense_df.groupby(['defteam', 'season'])['interception']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    defense_df['qb_hit_ewma'] = defense_df.groupby(['defteam', 'season'])['qb_hit']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    defense_df['sack_ewma'] = defense_df.groupby(['defteam', 'season'])['sack']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    defense_df['pass_touchdown_ewma'] = defense_df.groupby(['defteam', 'season'])['pass_touchdown']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    defense_df['passing_yards_ewma'] = defense_df.groupby(['defteam', 'season'])['passing_yards']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    defense_df['cpoe_ewma'] = defense_df.groupby(['defteam', 'season'])['cpoe']\
        .transform(lambda x: x.ewm(min_periods=1, span=5).mean())

    return defense_df

def update_week_to_next(df):
    """
    Function to update all teams to the next NFL week based on the current max week.
    Teams that are behind the max week will jump to the next NFL week.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing NFL data including the 'week' column.
    
    Returns:
    pd.DataFrame: DataFrame with 'week' values updated appropriately.
    """
    df_copy = df.copy()

    #Current week
    current_week = df_copy['week'].max()

    #Increment all weeks to the next NFL week (based on current max week)
    df_copy['week'] = df_copy['week'].apply(lambda x: current_week + 1)
    
    return df_copy

def predict_passing_yards(df, model, preprocessor, features, categorical_columns, numeric_columns, target_column='predicted_passing_yards'):
    """
    Function to make passing yard predictions on a new dataframe.
    
    Parameters:
    new_data (pd.DataFrame): The new dataframe containing feature columns.
    model (xgb.XGBRegressor): Trained XGBoost model.
    preprocessor (ColumnTransformer): Preprocessor object for transforming the features.
    features (list): List of feature columns used by the model.
    categorical_columns (list): List of categorical columns to be preprocessed.
    numeric_columns (list): List of numeric columns to be preprocessed.
    target_column (str): The name of the column for storing the predictions. Default is 'predicted_passing_yards'.
    
    Returns:
    pd.DataFrame: The input dataframe with an added column of predicted passing yards.
    """
    
    #Select and transform the features from the new data
    X_new = df[features]
    
    #Apply preprocessing (scaling and encoding)
    X_new_transformed = preprocessor.transform(X_new)
    
    #Make predictions using the trained model
    df[target_column] = model.predict(X_new_transformed)
    
    return df

def monte_carlo_simulations_truncated(df, global_residual_std, residual_col='pyoe_list',
                                      predicted_col='predicted_passing_yards', num_simulations=1000,
                                      noise_scale=0.25, min_yards=50, max_yards=650):
    """
    Monte Carlo simulation for all QBs using truncated normal distribution to ensure realistic passing yard values.
    :param df: DataFrame containing QBs with predicted means and mean team residuals
    :param global_residual_std: Global residual standard deviation to use for the distribution
    :param residual_col: Column containing team-specific mean residuals
    :param predicted_col: Column containing predicted passing yards for each QB
    :param num_simulations: Number of Monte Carlo simulations to run per QB
    :param noise_scale: Standard deviation of the added noise
    :param min_yards: Minimum passing yards to ensure no unrealistic low values
    :param max_yards: Maximum passing yards to cap extreme outliers
    :return: Dictionary of QBs with their simulated passing yard distributions
    """
    qb_simulations = {}

    for idx, row in df.iterrows():
        qb_name = row['qb_name']
        predicted_mean = row[predicted_col]
        mean_team_residual = row[residual_col]

        # Calculate the truncated normal distribution parameters
        lower_bound = (min_yards - predicted_mean) / global_residual_std
        upper_bound = (max_yards - predicted_mean) / global_residual_std

        # Sample from the truncated normal distribution
        truncated_simulations = truncnorm.rvs(lower_bound, upper_bound, loc=predicted_mean, scale=global_residual_std, size=num_simulations)

        # Add a small amount of noise to introduce variability
        noise = np.random.normal(loc=0, scale=noise_scale, size=num_simulations)
        simulations = truncated_simulations + noise

        # Ensure no negative values by applying the minimum passing yard threshold
        simulations = np.maximum(min_yards, simulations)

        # Save the simulations for this QB
        qb_simulations[qb_name] = simulations

    return qb_simulations


#Notebook Game Functions

def implied_odds_to_american(implied_prob):
    """
    Convert implied probability to American odds format.
    :param implied_prob: Implied probability (between 0 and 1)
    :return: American odds (with + for positive odds)
    """
    if implied_prob == 0:
        return 'N/A'
    if implied_prob >= 0.5:
        american_odds = -100 / implied_prob
    else:
        american_odds = (1 / implied_prob - 1) * 100
    
    # If positive odds, add the "+" sign in front
    if american_odds > 0:
        return f"+{round(american_odds)}"
    else:
        return round(american_odds)
    
def midpoint_madness(qb_name, qb_simulations, over_qbs=[], under_qbs=[]):
    """
    Function to simulate over/under passing yard predictions in Midpoint Madness game.
    
    :param qb_name: The name of the main QB to start with.
    :param qb_simulations: Dictionary containing simulated passing yard results for all QBs.
    :param over_qbs: List of QB names that the user thinks will pass for fewer yards than the main QB.
    :param under_qbs: List of QB names that the user thinks will pass for more yards than the main QB.
    
    :return: Dictionary of probabilities or betting odds for each over/under comparison.
    """
    results = {}
    combined_prob = 1  # To calculate the combined probability

    # Simulations for the chosen QB
    main_qb_simulations = qb_simulations[qb_name]
    
    # Calculate probabilities for 'over' comparisons
    for over_qb in over_qbs:
        over_qb_simulations = qb_simulations[over_qb]
        prob_main_qb_more = np.mean(main_qb_simulations > over_qb_simulations)
        results[f"{qb_name} > {over_qb}"] = {
            'probability': prob_main_qb_more,
            'american_odds': implied_odds_to_american(prob_main_qb_more)
        }
        # Update the combined probability
        combined_prob *= prob_main_qb_more
    
    # Calculate probabilities for 'under' comparisons
    for under_qb in under_qbs:
        under_qb_simulations = qb_simulations[under_qb]
        prob_main_qb_less = np.mean(main_qb_simulations < under_qb_simulations)
        results[f"{qb_name} < {under_qb}"] = {
            'probability': prob_main_qb_less,
            'american_odds': implied_odds_to_american(prob_main_qb_less)
        }
        # Update the combined probability
        combined_prob *= prob_main_qb_less
    
    # Add combined probability to the results
    results['combined_probability'] = combined_prob
    results['combined_american_odds'] = implied_odds_to_american(combined_prob)
    
    return results


#*****APP SPECIFIC***** note: this shouldn't be a thing, figure out after initial iteration
# Import your midpoint madness function

def implied_odds_to_american_app(probability, cap_large=True, cap_value=5000):
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

def midpoint_madness_app(qb_name, qb_simulations, over_qbs=[], under_qbs=[]):
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
            'american_odds': implied_odds_to_american_app(prob_main_qb_more)
        }
        # Update the combined probability
        combined_prob *= prob_main_qb_more
    
    # Calculate probabilities for 'under' comparisons
    for under_qb in under_qbs:
        under_qb_simulations = qb_simulations[under_qb]
        prob_main_qb_less = np.mean(np.array(main_qb_simulations) < np.array(under_qb_simulations))
        results[f"{qb_name} < {under_qb}"] = {
            'probability': prob_main_qb_less,
            'american_odds': implied_odds_to_american_app(prob_main_qb_less)
        }
        # Update the combined probability
        combined_prob *= prob_main_qb_less
    
    # Add combined probability to the results
    results['combined_probability'] = combined_prob
    results['combined_american_odds'] = implied_odds_to_american_app(combined_prob, cap_large=True)
    
    return results