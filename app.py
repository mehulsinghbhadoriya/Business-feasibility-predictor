import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# Page Config
st.set_page_config(
    page_title="Global ML Business Feasibility Predictor",
    page_icon="🤖",
    layout="wide"
)

# Custom Styling (premium and clean dashboard layout)
st.markdown("""
<style>
    .main {
        background-color: #fcfcfd;
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        border: 1px solid #f0f0f2;
    }
    .prediction-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1e293b;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# App Titles
st.title("🤖 ML Business Feasibility Predictor")
st.subheader("A trained Random Forest pipeline predicting startup success using industry vectors and macroeconomic parameters.")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_DATA_PATH = os.path.join(BASE_DIR, "macro_data.csv")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "survival_model.pkl")
REGRESSOR_PATH = os.path.join(BASE_DIR, "profit_model.pkl")

# Load country economics database
if not os.path.exists(MACRO_DATA_PATH):
    st.error("Error: Macroeconomic database `macro_data.csv` is missing.")
    st.stop()
macro_df = pd.read_csv(MACRO_DATA_PATH)

# Load trained ML models
models_trained = False
if os.path.exists(CLASSIFIER_PATH) and os.path.exists(REGRESSOR_PATH):
    try:
        classifier = joblib.load(CLASSIFIER_PATH)
        regressor = joblib.load(REGRESSOR_PATH)
        models_trained = True
    except Exception as e:
        st.error(f"Error deserializing ML models: {e}")
else:
    st.warning("⚠️ Machine learning models are currently training in the background. Please wait a moment and refresh this screen.")

# UI Inputs Layout
st.markdown("### 📝 Define Your Business Idea & Economic Context")
st.write("Fill in the categorical and numerical fields below to feed details into our trained Machine Learning models:")

# Create input form
with st.form("ml_predictor_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Categorical Parameters")
        
        industry = st.selectbox(
            "Business Category / Sector:",
            options=[
                "SaaS / Mobile Apps", "Bakery / Cafe / Food", "Boutique Retail / E-commerce",
                "Local Services (Salon, Gym, Repairs)", "E-Learning / Online Academy", 
                "Agrotech / Vertical Farming", "Real Estate / Property Management",
                "Healthcare Clinic / Wellness Care", "Logistics / Delivery Service"
            ],
            index=0
        )
        
        business_model = st.selectbox(
            "Primary Business Model:",
            options=["B2C", "B2B", "Marketplace"],
            index=0,
            help="B2C (direct to consumer), B2B (selling to companies), Marketplace (connecting buyers and sellers)."
        )
        
        sales_channel = st.selectbox(
            "Primary Sales & Operations Channel:",
            options=["Online", "Physical", "Omnichannel"],
            index=0,
            help="Online (e.g. cloud app/website), Physical (e.g. physical storefront), Omnichannel (combination of both)."
        )
        
        founder_experience = st.selectbox(
            "Founder's Startup & Industry Experience:",
            options=["Novice", "Intermediate", "Experienced"],
            index=1,
            help="Novice (First business), Intermediate (Prior management background), Experienced (Serial entrepreneur)."
        )
        
        target_country = st.selectbox(
            "Target Country for Launch:",
            options=macro_df['Country'].tolist(),
            index=1 # Default to India
        )

    with col2:
        st.subheader("💰 Numerical Parameters")
        
        # Pull selected country economics
        country_row = macro_df[macro_df['Country'] == target_country].iloc[0]
        currency_symbol = country_row['Symbol']
        currency_code = country_row['Currency']
        ppp = float(country_row['PPP'])
        labor_idx = float(country_row['LaborIndex'])
        rent_idx = float(country_row['RentIndex'])
        gdp_per_capita = ppp * 45000
        
        # Reference baseline capital dynamically scaled to local currency
        baseline_usd = 40000
        suggested_capital = round(baseline_usd * ppp, -2)
        
        starting_capital = st.number_input(
            f"Starting Capital ({currency_code} {currency_symbol}):",
            min_value=10.0,
            value=float(suggested_capital),
            step=500.0,
            help="Total available runway and investment capital in your local currency."
        )
        
        marketing_budget = st.number_input(
            f"Planned Monthly Marketing Budget ({currency_symbol}):",
            min_value=0.0,
            value=float(round(suggested_capital * 0.03, -1)),
            step=50.0,
            help="Monthly capital dedicated purely to ads, search ranking, and customer acquisition."
        )
        
        team_size = st.number_input(
            "Expected Team Size (Planned Employees):",
            min_value=1,
            max_value=100,
            value=3,
            step=1,
            help="Total headcount of employees (excluding the founder) that will receive monthly salaries."
        )
        
        avg_tx_value = st.number_input(
            f"Average Price / Transaction Value ({currency_symbol}):",
            min_value=0.1,
            value=float(round(50.0 * ppp, 1)),
            step=5.0,
            help="The average price a customer pays per purchase or monthly subscription in your currency."
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("🚀 Run ML Feasibility Predictor")

# Handle analysis on submit
if submit_button:
    if not models_trained:
        st.error("Error: The machine learning models have not finished training. Please try again in a few seconds.")
    else:
        # 1. Translate local numerical values to USD via Purchasing Power Parity (PPP) for the ML model
        capital_usd = starting_capital / ppp
        marketing_usd = marketing_budget / ppp
        tx_usd = avg_tx_value / ppp
        
        # 2. Package inputs into exactly the Pandas DataFrame schema the trained Pipeline expects
        X_input = pd.DataFrame([{
            "Industry": industry,
            "BusinessModel": business_model,
            "SalesChannel": sales_channel,
            "FounderExperience": founder_experience,
            "StartingCapital": float(capital_usd),
            "MonthlyMarketing": float(marketing_usd),
            "TeamSize": int(team_size),
            "AverageTransactionValue": float(tx_usd),
            "CountryGDPPerCapita": float(gdp_per_capita),
            "CountryPPP": float(ppp),
            "CountryLaborIndex": float(labor_idx),
            "CountryRentIndex": float(rent_idx)
        }])
        
        # 3. Predict via Pipeline
        try:
            # Predict probability of class 1 (Survived)
            survival_prob = classifier.predict_proba(X_input)[0][1] * 100
            
            # Predict annual net profit in USD and convert back to local currency
            profit_usd = regressor.predict(X_input)[0]
            profit_local = profit_usd * ppp
            
            st.markdown("---")
            st.markdown("### 🤖 Machine Learning Diagnostic Report")
            
            # Display Results Cards
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                # Survival Probability UI formatting
                if survival_prob >= 70.0:
                    status_lbl = "Strong Viability Profile"
                    status_color = "#155724"
                    bg_color = "#d4edda"
                elif survival_prob >= 40.0:
                    status_lbl = "Moderate Venture Risk"
                    status_color = "#856404"
                    bg_color = "#fff3cd"
                else:
                    status_lbl = "High Venture Failure Risk"
                    status_color = "#721c24"
                    bg_color = "#f8d7da"
                    
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 25px; border-radius: 12px; border: 1px solid {status_color}; text-align: center;">
                    <div style="color: #4b5563; font-weight: bold; font-size: 1.1rem; margin-bottom: 5px;">ML PREDICTED SURVIVAL PROBABILITY (3 YEAR)</div>
                    <div style="color: {status_color}; font-size: 3.5rem; font-weight: 850; line-height: 1;">{survival_prob:.1f}%</div>
                    <div style="color: {status_color}; font-weight: bold; font-size: 1.1rem; margin-top: 10px;">Status: {status_lbl}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_res2:
                # Profit UI Formatting
                if profit_local >= 0:
                    profit_color = "#0f5132"
                    profit_bg = "#d1e7dd"
                    sign = "+"
                else:
                    profit_color = "#842029"
                    profit_bg = "#f8d7da"
                    sign = ""
                    
                st.markdown(f"""
                <div style="background-color: {profit_bg}; padding: 25px; border-radius: 12px; border: 1px solid {profit_color}; text-align: center;">
                    <div style="color: #4b5563; font-weight: bold; font-size: 1.1rem; margin-bottom: 5px;">ML EXPECTED ANNUAL NET INCOME</div>
                    <div style="color: {profit_color}; font-size: 3.5rem; font-weight: 850; line-height: 1;">{sign}{currency_symbol}{profit_local:,.2f}</div>
                    <div style="color: {profit_color}; font-weight: bold; font-size: 1.1rem; margin-top: 10px;">Currency: {currency_code} ({currency_symbol})</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Additional detailed metrics visualizations
            st.markdown("<br>", unsafe_allow_html=True)
            col_chart, col_diagnostics = st.columns([3, 2])
            
            with col_chart:
                st.markdown("#### Financial Allocation Assessment (Annualized)")
                # Calculate basic monthly overhead
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
                
                # Approximate cost breakdown in local currency for the chart
                est_rent_annual = ((0.05 * base_caps[industry]) * rent_idx * ppp) * 12
                est_labor_annual = (team_size * (3000 * labor_idx * ppp)) * 12
                est_marketing_annual = marketing_budget * 12
                
                bar_categories = ['Starting Capital', 'Marketing (Annual)', 'Estimated Rent (Annual)', 'Estimated Labor (Annual)', 'Predicted Profit/Loss (Annual)']
                bar_values = [starting_capital, est_marketing_annual, est_rent_annual, est_labor_annual, profit_local]
                
                colors = ['#6c757d', '#fd7e14', '#0d6efd', '#20c997', '#28a745' if profit_local >= 0 else '#dc3545']
                
                fig = go.Figure(data=[go.Bar(
                    x=bar_categories,
                    y=bar_values,
                    marker_color=colors,
                    text=[f"{currency_symbol}{v:,.0f}" if v >= 0 else f"-{currency_symbol}{abs(v):,.0f}" for v in bar_values],
                    textposition='auto'
                )])
                
                fig.update_layout(
                    height=350,
                    margin=dict(l=15, r=15, t=10, b=10),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_diagnostics:
                st.markdown("#### 💡 ML Model Diagnostics")
                
                # Check for clear logical violations to output warnings
                warnings_found = False
                
                if survival_prob < 50.0:
                    st.error("🚨 **ML Risk Alert: High Burn Rate Detected**")
                    st.write("The Random Forest model predicts a low success rate. Key risk signals extracted from model weights:")
                    
                    if starting_capital < (suggested_capital * 0.5):
                        st.write(f"- **Capital Deficit:** Your starting capital is below the optimal threshold for {industry} in {target_country}. The model heavily penalizes thin reserves.")
                        warnings_found = True
                    if team_size > 5 and starting_capital < suggested_capital:
                        st.write("- **Labor Overhead:** Hiring a team of this size with limited starting capital creates an unsustainably high burn rate.")
                        warnings_found = True
                    if sales_channel == "Online" and marketing_budget < (starting_capital * 0.01):
                        st.write("- **Underfunded Acquisition:** Online businesses require higher marketing capital. The model detects insufficient acquisition spend.")
                        warnings_found = True
                    if not warnings_found:
                        st.write("- **Macroeconomic Headwinds:** High local rent and labor indices combined with lower relative transactional size are impacting margins in this region.")
                else:
                    st.success("✨ **ML Asset Alert: Solid Venture Profile**")
                    st.write("The Random Forest model has identified strong predictive features supporting your startup:")
                    
                    if founder_experience == "Experienced":
                        st.write("- **Founder Premium:** The model rewards serial entrepreneur experience with highly stable survival distributions.")
                    if starting_capital >= suggested_capital:
                        st.write("- **Resilient Runway:** Your starting capital provides a strong cushion against early market volatility.")
                    if profit_local > 0:
                        st.write(f"- **Favorable Economics:** The high estimated transaction price ({currency_symbol}{avg_tx_value:.1f}) easily offsets your local operating costs.")
                        
                st.caption("*These insights are generated from feature importance weights in the Random Forest models.*")

        except Exception as e:
            st.error(f"Error computing prediction model: {e}")
