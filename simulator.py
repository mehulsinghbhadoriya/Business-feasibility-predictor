import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants/Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDUSTRY_DATA_PATH = os.path.join(BASE_DIR, "industry_data.json")
MACRO_DATA_PATH = os.path.join(BASE_DIR, "macro_data.csv")

def load_databases():
    """Loads industry benchmark data and global macroeconomic CSV."""
    if not os.path.exists(INDUSTRY_DATA_PATH):
        raise FileNotFoundError(f"Missing {INDUSTRY_DATA_PATH}")
    if not os.path.exists(MACRO_DATA_PATH):
        raise FileNotFoundError(f"Missing {MACRO_DATA_PATH}")
        
    with open(INDUSTRY_DATA_PATH, "r") as f:
        industries = json.load(f)
        
    macro_df = pd.read_csv(MACRO_DATA_PATH)
    return industries, macro_df

def match_business_idea(user_idea_text):
    """
    Computes TF-IDF vector similarity between user text input and industry keywords.
    Returns matched industry dict and confidence score.
    """
    industries, _ = load_databases()
    
    # Extract corpus of keywords
    corpus = [ind["keywords"] for ind in industries]
    
    # Prepend user idea to corpus to vectorize all at once
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = [user_idea_text] + corpus
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Compute cosine similarities between user vector and industry vectors
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    # If confidence is extremely low, fall back or default
    matched_industry = industries[best_match_idx]
    
    return matched_industry, float(best_score)

def scale_finances(industry, macro_row):
    """
    Applies Purchasing Power Parity (PPP), labor cost indexes, and commercial 
    rent indexes to scale US baseline costs/revenues to the destination country.
    """
    # Force float conversion
    ppp = float(macro_row['PPP'])
    labor_idx = float(macro_row['LaborIndex'])
    rent_idx = float(macro_row['RentIndex'])
    
    # Calculate PPP localized metrics
    scaled = {
        "startup_capital": round(industry['base_startup_capital'] * ppp, -2), # round to nearest hundred
        "monthly_rent": round(industry['base_monthly_rent'] * rent_idx * ppp, -1), # round to nearest ten
        "monthly_labor": round(industry['base_monthly_labor'] * labor_idx * ppp, -1),
        "monthly_ops": round(industry['base_monthly_ops'] * ppp, -1),
        "monthly_revenue": round(industry['base_monthly_revenue'] * ppp, -1)
    }
    
    # Avoid zero costs for non-free baselines
    if industry['base_monthly_rent'] > 0 and scaled['monthly_rent'] == 0:
        scaled['monthly_rent'] = 10.0
    if industry['base_monthly_labor'] > 0 and scaled['monthly_labor'] == 0:
        scaled['monthly_labor'] = 10.0
        
    return scaled

def run_monte_carlo(startup_capital, monthly_revenue, rent, labor, ops, rev_vol, cost_vol, months=12, runs=2000):
    """
    Executes a stochastic Monte Carlo simulation modeling dynamic month-over-month cash flows.
    Tracks net cash balances, bankruptcy probabilities, and outputs distributional percentiles.
    """
    fixed_costs = rent + labor + ops
    
    # Cash trajectory matrix: rows = simulation runs, columns = months + 1 (starting capital at index 0)
    cash_trajectories = np.zeros((runs, months + 1))
    cash_trajectories[:, 0] = startup_capital
    
    # Survival vector: True if run stays positive, False if it bankrupts
    survived = np.ones(runs, dtype=bool)
    bankruptcy_month = np.zeros(runs)
    
    for month in range(1, months + 1):
        # Generate random, normally distributed revenues and costs for all runs
        # Revenue cannot be negative
        sim_revenues = np.random.normal(monthly_revenue, rev_vol * monthly_revenue, runs)
        sim_revenues = np.maximum(0, sim_revenues)
        
        sim_costs = np.random.normal(fixed_costs, cost_vol * fixed_costs, runs)
        sim_costs = np.maximum(0, sim_costs)
        
        # Calculate monthly cash change
        cash_changes = sim_revenues - sim_costs
        
        # Update cash balances for currently active runs
        for run_idx in range(runs):
            if survived[run_idx]:
                prev_cash = cash_trajectories[run_idx, month - 1]
                new_cash = prev_cash + cash_changes[run_idx]
                
                if new_cash <= 0:
                    survived[run_idx] = False
                    bankruptcy_month[run_idx] = month
                    cash_trajectories[run_idx, month:] = 0.0 # Stays at zero once bankrupt
                else:
                    cash_trajectories[run_idx, month] = new_cash
            else:
                cash_trajectories[run_idx, month] = 0.0
                
    # Calculate key output metrics
    survival_probability = float(np.sum(survived) / runs * 100)
    
    # Extract end-of-simulation stats for surviving runs
    final_balances = cash_trajectories[:, -1]
    
    # Get percentiles of final balances across all runs to show outcomes distribution
    p10_worst_case = float(np.percentile(final_balances, 10))
    p50_median_case = float(np.percentile(final_balances, 50))
    p90_best_case = float(np.percentile(final_balances, 90))
    
    # Compute median cash trajectory across all runs for visualization
    median_trajectory = np.percentile(cash_trajectories, 50, axis=0).tolist()
    p10_trajectory = np.percentile(cash_trajectories, 10, axis=0).tolist()
    p90_trajectory = np.percentile(cash_trajectories, 90, axis=0).tolist()
    
    # Average break-even indicators: month where net cash was positive
    # We find how many months on average runs operated profitably
    avg_months_runway = 12
    if not np.all(survived):
        avg_months_runway = float(np.mean(bankruptcy_month[~survived]))
        
    return {
        "survival_probability": round(survival_probability, 1),
        "worst_case_balance": round(p10_worst_case, 2),
        "median_case_balance": round(p50_median_case, 2),
        "best_case_balance": round(p90_best_case, 2),
        "median_trajectory": median_trajectory,
        "p10_trajectory": p10_trajectory,
        "p90_trajectory": p90_trajectory,
        "avg_months_runway": round(avg_months_runway, 1)
    }
