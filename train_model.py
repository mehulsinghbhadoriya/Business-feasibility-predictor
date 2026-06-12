import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_DATA_PATH = os.path.join(BASE_DIR, "macro_data.csv")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "survival_model.pkl")
REGRESSOR_PATH = os.path.join(BASE_DIR, "profit_model.pkl")

def generate_historical_dataset(macro_df, n_samples=5000):
    """
    Synthesizes a high-fidelity dataset of 5,000 historical business launches 
    incorporating real-world macroeconomic and startup operational correlations.
    """
    np.random.seed(42)
    
    # Categorical option spaces
    industries = [
        "SaaS / Mobile Apps", "Bakery / Cafe / Food", "Boutique Retail / E-commerce",
        "Local Services (Salon, Gym, Repairs)", "E-Learning / Online Academy", 
        "Agrotech / Vertical Farming", "Real Estate / Property Management",
        "Healthcare Clinic / Wellness Care", "Logistics / Delivery Service"
    ]
    models = ["B2B", "B2C", "Marketplace"]
    channels = ["Online", "Physical", "Omnichannel"]
    experiences = ["Novice", "Intermediate", "Experienced"]
    
    # Baseline capital requirements per industry (USD)
    industry_baselines = {
        "SaaS / Mobile Apps": 15000,
        "Bakery / Cafe / Food": 80000,
        "Boutique Retail / E-commerce": 50000,
        "Local Services (Salon, Gym, Repairs)": 25000,
        "E-Learning / Online Academy": 8000,
        "Agrotech / Vertical Farming": 120000,
        "Real Estate / Property Management": 60000,
        "Healthcare Clinic / Wellness Care": 100000,
        "Logistics / Delivery Service": 90000
    }
    
    # Pre-select random rows from macro_df to match location demographics
    selected_countries = macro_df.sample(n_samples, replace=True).reset_index(drop=True)
    
    data = []
    for i in range(n_samples):
        # 1. Randomly assign user-input parameters
        ind = np.random.choice(industries)
        model = np.random.choice(models)
        channel = np.random.choice(channels)
        exp = np.random.choice(experiences)
        
        # Pull location macro indicators
        country_row = selected_countries.iloc[i]
        ppp = float(country_row['PPP'])
        labor_idx = float(country_row['LaborIndex'])
        rent_idx = float(country_row['RentIndex'])
        gdp_factor = ppp * 45000 # Estimate GDP per capita based on PPP scaling
        
        # 2. Assign numerical parameters with realistic ranges relative to country scales
        base_cap = industry_baselines[ind] * ppp
        # Generate startup capital around the baseline (e.g. from 20% to 300% of baseline)
        capital = np.random.exponential(scale=base_cap * 1.2)
        capital = np.clip(capital, base_cap * 0.1, base_cap * 10.0)
        
        # Monthly marketing relative to capital (0.5% to 8%)
        marketing = np.random.uniform(0.005, 0.08) * capital
        
        # Team Size based on industry and model
        if ind in ["SaaS / Mobile Apps", "Agrotech / Vertical Farming", "Logistics / Delivery Service"]:
            team_size = int(np.random.randint(1, 12))
        elif ind in ["Local Services (Salon, Gym, Repairs)", "Bakery / Cafe / Food"]:
            team_size = int(np.random.randint(2, 8))
        else:
            team_size = int(np.random.randint(1, 5))
            
        # Average Transaction Value based on business model
        if model == "B2B":
            tx_val = np.random.uniform(100.0, 3000.0) * ppp
        else:
            tx_val = np.random.uniform(10.0, 150.0) * ppp
            
        # 3. Apply business economics scoring logic to determine "Survived" label
        score = 0.5 # Neutral baseline
        
        # Capital Factor (High capital = safety, Low capital = high risk)
        cap_ratio = capital / base_cap
        score += np.log10(cap_ratio) * 0.5
        
        # Operating burn cost calculation (Rent + Staff)
        monthly_rent = (0.05 * base_cap) * rent_idx
        monthly_labor = team_size * (3000 * labor_idx * ppp)
        monthly_ops = (0.02 * base_cap)
        monthly_burn = monthly_rent + monthly_labor + monthly_ops
        
        # High burn rate relative to capital is dangerous
        if capital > 0:
            burn_ratio = monthly_burn / capital
            if burn_ratio > 0.15: # Runway less than 7 months
                score -= 0.4
            elif burn_ratio < 0.05: # Runway greater than 20 months
                score += 0.2
                
        # Marketing Factor: Online channels depend heavily on marketing
        if channel == "Online":
            mktg_to_burn_ratio = marketing / max(100.0, monthly_burn)
            if mktg_to_burn_ratio < 0.02:
                score -= 0.3 # Underfunded marketing for online businesses leads to failure
            elif mktg_to_burn_ratio > 0.10:
                score += 0.15
                
        # Experience Factor
        if exp == "Experienced":
            score += 0.25
        elif exp == "Novice":
            score -= 0.15
            
        # Channel-Industry Fit (e.g. Cafe cannot be online only)
        if ind == "Bakery / Cafe / Food" and channel == "Online":
            score -= 0.6
        if ind == "Healthcare Clinic / Wellness Care" and channel == "Online":
            score -= 0.5
        if ind == "SaaS / Mobile Apps" and channel == "Physical":
            score -= 0.4
            
        # Calculate survival probability from score (Sigmoid)
        survival_prob = 1.0 / (1.0 + np.exp(-3.5 * score))
        survived = 1 if np.random.random() < survival_prob else 0
        
        # Calculate Expected Annual Net Profit/Loss based on inputs and survival status
        # Profit is highly correlated with capital scale and survival outcome
        base_revenue = (monthly_burn * 1.15) + (np.random.normal(0, 0.2) * monthly_burn)
        
        if survived == 1:
            annual_profit = (base_revenue - monthly_burn) * 12
            annual_profit += np.random.normal(0, 0.1 * abs(annual_profit))
        else:
            # Failed businesses incur heavy capital losses (between 50% and 100% of capital lost)
            annual_profit = -np.random.uniform(0.5, 1.0) * capital
            
        data.append({
            "Industry": ind,
            "BusinessModel": model,
            "SalesChannel": channel,
            "FounderExperience": exp,
            "StartingCapital": round(capital, 2),
            "MonthlyMarketing": round(marketing, 2),
            "TeamSize": int(team_size),
            "AverageTransactionValue": round(tx_val, 2),
            "CountryGDPPerCapita": round(gdp_factor, 2),
            "CountryPPP": round(ppp, 2),
            "CountryLaborIndex": round(labor_idx, 2),
            "CountryRentIndex": round(rent_idx, 2),
            "Survived": int(survived),
            "AnnualProfit": round(annual_profit, 2)
        })
        
    return pd.DataFrame(data)

def train_and_save_pipelines():
    print("Step 1: Loading global macro CSV for training context...")
    if not os.path.exists(MACRO_DATA_PATH):
        raise FileNotFoundError(f"Macro data missing at: {MACRO_DATA_PATH}")
    macro_df = pd.read_csv(MACRO_DATA_PATH)
    
    print("Step 2: Generating synthetic historical database of 5,000 launches...")
    df = generate_historical_dataset(macro_df, n_samples=5000)
    
    # Separate Features and Targets
    X = df.drop(columns=["Survived", "AnnualProfit"])
    y_class = df["Survived"]
    y_reg = df["AnnualProfit"]
    
    # Identify preprocessing groups
    categorical_cols = ["Industry", "BusinessModel", "SalesChannel", "FounderExperience"]
    numeric_cols = [
        "StartingCapital", "MonthlyMarketing", "TeamSize", "AverageTransactionValue", 
        "CountryGDPPerCapita", "CountryPPP", "CountryLaborIndex", "CountryRentIndex"
    ]
    
    print("Step 3: Building preprocessor pipeline stages...")
    # Column Transformer handles encoding and standard scaling perfectly
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ]
    )
    
    # Construct complete unified pipeline objects
    classifier_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42))
    ])
    
    regressor_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42))
    ])
    
    print("Step 4: Training Random Forest Classifier (Survival Likelihood)...")
    classifier_pipeline.fit(X, y_class)
    
    print("Step 5: Training Random Forest Regressor (Expected Annual Profit)...")
    regressor_pipeline.fit(X, y_reg)
    
    print("Step 6: Serializing and exporting pipelines to local files...")
    joblib.dump(classifier_pipeline, CLASSIFIER_PATH)
    joblib.dump(regressor_pipeline, REGRESSOR_PATH)
    
    print("==================================================")
    print("SUCCESS: ML pipelines successfully trained and saved!")
    print("Classifier exported:", CLASSIFIER_PATH)
    print("Regressor exported:", REGRESSOR_PATH)
    print("==================================================")

if __name__ == "__main__":
    train_and_save_pipelines()
