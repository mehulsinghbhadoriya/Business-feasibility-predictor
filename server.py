import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI App
app = FastAPI(title="Global Business Feasibility API", version="1.0.0")

# Enable CORS for React frontend communications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_DATA_PATH = os.path.join(BASE_DIR, "macro_data.csv")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "survival_model.pkl")
REGRESSOR_PATH = os.path.join(BASE_DIR, "profit_model.pkl")

# Load Macroeconomic Database
if not os.path.exists(MACRO_DATA_PATH):
    raise FileNotFoundError("Missing macro_data.csv")
macro_df = pd.read_csv(MACRO_DATA_PATH)

# Load Trained ML Models
if os.path.exists(CLASSIFIER_PATH) and os.path.exists(REGRESSOR_PATH):
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
else:
    classifier, regressor = None, None
    print("WARNING: ML Models not found. Run train_model.py first.")

# Baseline capital requirements per industry (USD)
base_caps = {
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

class PredictionRequest(BaseModel):
    industry: str
    business_model: str
    sales_channel: str
    founder_experience: str
    starting_capital: float
    marketing_budget: float
    team_size: int
    avg_tx_value: float
    country_code: str

@app.get("/api/countries")
def get_countries():
    """Returns supported countries and their macroeconomic indicators."""
    countries = []
    for _, row in macro_df.iterrows():
        countries.append({
            "name": row["Country"],
            "code": row["CountryCode"],
            "currency": row["Currency"],
            "symbol": row["Symbol"],
            "ppp": float(row["PPP"]),
            "labor_index": float(row["LaborIndex"]),
            "rent_index": float(row["RentIndex"])
        })
    return countries

@app.post("/api/predict")
def predict(payload: PredictionRequest):
    """
    Translates local monetary inputs using PPP, runs pre-trained Random Forest pipelines,
    and returns localized predictions and diagnostics.
    """
    if classifier is None or regressor is None:
        raise HTTPException(status_code=503, detail="Machine learning models are not loaded on server.")
        
    # 1. Resolve Country Economic Indicators
    country_rows = macro_df[macro_df["CountryCode"] == payload.country_code]
    if country_rows.empty:
        raise HTTPException(status_code=400, detail="Invalid country code provided.")
    country_row = country_rows.iloc[0]
    
    ppp = float(country_row["PPP"])
    labor_idx = float(country_row["LaborIndex"])
    rent_idx = float(country_row["RentIndex"])
    gdp_per_capita = ppp * 45000
    
    # 2. Scale local currencies to USD standard features for ML pipeline
    capital_usd = payload.starting_capital / ppp
    marketing_usd = payload.marketing_budget / ppp
    tx_usd = payload.avg_tx_value / ppp
    
    # 3. Package as Pandas DataFrame matching Pipeline schema
    X_input = pd.DataFrame([{
        "Industry": payload.industry,
        "BusinessModel": payload.business_model,
        "SalesChannel": payload.sales_channel,
        "FounderExperience": payload.founder_experience,
        "StartingCapital": float(capital_usd),
        "MonthlyMarketing": float(marketing_usd),
        "TeamSize": int(payload.team_size),
        "AverageTransactionValue": float(tx_usd),
        "CountryGDPPerCapita": float(gdp_per_capita),
        "CountryPPP": float(ppp),
        "CountryLaborIndex": float(labor_idx),
        "CountryRentIndex": float(rent_idx)
    }])
    
    # 4. Predict Outcomes
    try:
        # Survival Probability
        survival_prob = float(classifier.predict_proba(X_input)[0][1] * 100)
        
        # Expected Net Profit (USD -> Local)
        profit_usd = float(regressor.predict(X_input)[0])
        profit_local = float(profit_usd * ppp)
        
        # Calculate local annualized cost breakdowns
        est_rent_annual = float(((0.05 * base_caps[payload.industry]) * rent_idx * ppp) * 12)
        est_labor_annual = float((payload.team_size * (3000 * labor_idx * ppp)) * 12)
        est_marketing_annual = float(payload.marketing_budget * 12)
        
        # 5. Extract diagnostics
        diagnostics = []
        suggested_capital = base_caps[payload.industry] * ppp
        
        if survival_prob < 50.0:
            if payload.starting_capital < (suggested_capital * 0.5):
                diagnostics.append(f"Capital Deficit: Starting capital is significantly below the optimal {country_row['Symbol']}{suggested_capital:,.0f} baseline for this sector.")
            if payload.team_size > 5 and payload.starting_capital < suggested_capital:
                diagnostics.append("High Labor Burn: Team size is unsustainably large for your starting cash reserves.")
            if payload.sales_channel == "Online" and payload.marketing_budget < (payload.starting_capital * 0.01):
                diagnostics.append("Underfunded Marketing: Online ventures depend heavily on active marketing acquisition. Your budget is below model recommendation.")
            if not diagnostics:
                diagnostics.append("Macroeconomic Stressors: High local rent/labor indices combined with thin margins are causing high failure risks.")
        else:
            if payload.founder_experience == "Experienced":
                diagnostics.append("Founder Experience Cushion: Your serial entrepreneur background lowers early operational failure risk.")
            if payload.starting_capital >= suggested_capital:
                diagnostics.append("Secure Financial Runway: Starting capital provides a solid buffer against early sales dry periods.")
            if profit_local > 0:
                diagnostics.append(f"Favorable Unit Economics: Average pricing is sufficient to cover local operational overhead.")
                
        return {
            "survival_probability": round(survival_prob, 1),
            "expected_annual_profit": round(profit_local, 2),
            "suggested_capital": round(suggested_capital, -2),
            "breakdown": {
                "rent_annual": round(est_rent_annual, 2),
                "labor_annual": round(est_labor_annual, 2),
                "marketing_annual": round(est_marketing_annual, 2)
            },
            "diagnostics": diagnostics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
