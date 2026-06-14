import React, { useState, useEffect } from 'react';

const INDUSTRIES = [
  { name: "SaaS / Mobile Apps", desc: "Software, Cloud & Tech Startups" },
  { name: "Bakery / Cafe / Food", desc: "Restaurants, Coffee & Eateries" },
  { name: "Boutique Retail / E-commerce", desc: "Apparel, Gifts & Online Stores" },
  { name: "Local Services (Salon, Gym, Repairs)", desc: "Salons, Fitness, Trades & Spas" },
  { name: "E-Learning / Online Academy", desc: "Tutoring, Courses & Education" },
  { name: "Agrotech / Vertical Farming", desc: "Hydroponics, Crop Tech & Urban Farms" },
  { name: "Real Estate / Property Management", desc: "Brokerage, Letting & Property Services" },
  { name: "Healthcare Clinic / Wellness Care", desc: "Dental, Physio, Spa & Therapy Clinics" },
  { name: "Logistics / Delivery Service", desc: "Last-mile courier, Trucking & Freight" }
];

export default function App() {
  // Navigation & Loading States
  const [activeStep, setActiveStep] = useState(0);
  const [countries, setCountries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [prediction, setPrediction] = useState(null);

  // Form Parameters State
  const [industry, setIndustry] = useState(INDUSTRIES[0].name);
  const [businessModel, setBusinessModel] = useState('B2C');
  const [salesChannel, setSalesChannel] = useState('Online');
  const [founderExperience, setFounderExperience] = useState('Intermediate');
  const [countryCode, setCountryCode] = useState('IN');
  const [startingCapital, setStartingCapital] = useState(10000);
  const [marketingBudget, setMarketingBudget] = useState(500);
  const [teamSize, setTeamSize] = useState(2);
  const [avgTxValue, setAvgTxValue] = useState(50);

  useEffect(() => {
    fetch('https://business-feasibility-predictor.onrender.com/api/countries')
      .then((res) => {
        if (!res.ok) throw new Error('API server is offline');
        return res.json();
      })
      .then((data) => {
        setCountries(data);
        if (data.length > 0) {
          // Default country to India or first available
          const defaultC = data.find(c => c.code === 'IN') || data[0];
          setCountryCode(defaultC.code);
          // Scale default starting capital recommendation
          setStartingCapital(defaultC.ppp * 40000);
          setMarketingBudget(defaultC.ppp * 40000 * 0.03);
          setAvgTxValue(Math.round(50 * defaultC.ppp));
        }
      })
      .catch((err) => {
        console.error(err);
        setError('Could not connect to the ML Backend. Ensure server.py is running on port 8000.');
      });
  }, []);

  // Update localized defaults when country selection changes
  const handleCountryChange = (code) => {
    setCountryCode(code);
    const selected = countries.find(c => c.code === code);
    if (selected) {
      setStartingCapital(Math.round(selected.ppp * 40000));
      setMarketingBudget(Math.round(selected.ppp * 40000 * 0.03));
      setAvgTxValue(Math.round(50 * selected.ppp));
    }
  };

  // Resolve active country details
  const activeCountry = countries.find(c => c.code === countryCode) || {
    symbol: '$', currency: 'USD', ppp: 1.0, name: 'United States'
  };

  const handlePredict = (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const payload = {
      industry,
      business_model: businessModel,
      sales_channel: salesChannel,
      founder_experience: founderExperience,
      starting_capital: parseFloat(startingCapital),
      marketing_budget: parseFloat(marketingBudget),
      team_size: parseInt(teamSize),
      avg_tx_value: parseFloat(avgTxValue),
      country_code: countryCode
    };

    fetch('https://business-feasibility-predictor.onrender.com/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then((res) => {
        if (!res.ok) throw new Error('Prediction request failed.');
        return res.json();
      })
      .then((data) => {
        setPrediction(data);
        setActiveStep(2); // Progress to report step
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Error communicating with server.');
        setLoading(false);
      });
  };

  // Determine indicator styles based on survival rate
  const getSurvivalStatus = (rate) => {
    if (rate >= 70.0) return { label: 'Strong Viability', color: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', pctColor: '#34d399', border: 'rgba(16, 185, 129, 0.3)' };
    if (rate >= 40.0) return { label: 'Moderate Risk', color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)', pctColor: '#fbbf24', border: 'rgba(245, 158, 11, 0.3)' };
    return { label: 'High Venture Risk', color: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)', pctColor: '#f87171', border: 'rgba(239, 68, 68, 0.3)' };
  };

  return (
    <div className="app-container">
      {/* Navbar Banner */}
      <header className="app-header">
        <div className="header-brand">
          <span className="logo-icon">🤖</span>
          <div>
            <h1>ScalePredict AI</h1>
            <p>Venture Feasibility & Market Modeling Engine</p>
          </div>
        </div>
        
        {/* Step Progress Tracker */}
        <div className="step-tracker">
          <div className={`step-node ${activeStep >= 0 ? 'active' : ''}`}>
            <span className="node-num">1</span>
            <span className="node-label">Core Concept</span>
          </div>
          <div className="step-line"></div>
          <div className={`step-node ${activeStep >= 1 ? 'active' : ''}`}>
            <span className="node-num">2</span>
            <span className="node-label">Financials & Target</span>
          </div>
          <div className="step-line"></div>
          <div className={`step-node ${activeStep >= 2 ? 'active' : ''}`}>
            <span className="node-num">3</span>
            <span className="node-label">ML Diagnosis</span>
          </div>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => setError('')} className="close-btn">&times;</button>
        </div>
      )}

      <main className="main-content">
        {loading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Querying pre-trained Random Forest pipelines...</p>
          </div>
        )}

        {/* Step 1: Core Concept Form with Split Hero Layout */}
        {activeStep === 0 && (
          <div className="form-step-panel fade-in split-layout">
            {/* Left Column: Hero Intro Banner */}
            <div className="step-hero-pane">
              <span className="hero-badge">MACHINE LEARNING SIMULATOR</span>
              <h2>Validate Your Next Business Venture Globally</h2>
              <p>
                Understand launch metrics, operational burn, and survival probabilities in seconds. 
                ScalePredict AI processes your business coordinates against 5,000+ historical startup 
                outcomes scaled to local purchasing power parameters.
              </p>
              
              <ul className="hero-features-list">
                <li>
                  <span className="feature-icon">🌍</span>
                  <div>
                    <strong>Macro Localization Engine</strong>
                    <p>Adjusts rent, labor overhead, and capital thresholds automatically using World Bank PPP conversion indices.</p>
                  </div>
                </li>
                <li>
                  <span className="feature-icon">📊</span>
                  <div>
                    <strong>Stochastic Capital Modeling</strong>
                    <p>Estimates early cash depletion thresholds and monthly operating burn safety margins.</p>
                  </div>
                </li>
                <li>
                  <span className="feature-icon">💡</span>
                  <div>
                    <strong>Pre-trained Classifier</strong>
                    <p>Uses a Random Forest algorithm to predict survival and generate custom risk diagnostics from model feature weights.</p>
                  </div>
                </li>
              </ul>
            </div>

            {/* Right Column: Configuration Controls */}
            <div className="step-config-pane">
              <h3>Venture Configuration</h3>
              <p className="section-desc">Select your target industry category and business structure:</p>
              
              <div className="industry-grid">
                {INDUSTRIES.map((ind) => (
                  <div 
                    key={ind.name}
                    className={`industry-card ${industry === ind.name ? 'selected' : ''}`}
                    onClick={() => setIndustry(ind.name)}
                  >
                    <div className="card-bullet"></div>
                    <h4>{ind.name}</h4>
                    <p>{ind.desc}</p>
                  </div>
                ))}
              </div>

              <div className="input-group-row">
                <div className="form-group">
                  <label>Business Model Archetype</label>
                  <select value={businessModel} onChange={(e) => setBusinessModel(e.target.value)}>
                    <option value="B2C">B2C (Business-to-Consumer)</option>
                    <option value="B2B">B2B (Business-to-Business)</option>
                    <option value="Marketplace">Marketplace / Hybrid</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Sales & Operations Channel</label>
                  <select value={salesChannel} onChange={(e) => setSalesChannel(e.target.value)}>
                    <option value="Online">Online / Digital Platform</option>
                    <option value="Physical">Physical Storefront / Office</option>
                    <option value="Omnichannel">Omnichannel / Hybrid</option>
                  </select>
                </div>
              </div>

              <div className="button-footer">
                <button 
                  type="button" 
                  className="btn btn-primary" 
                  onClick={() => setActiveStep(1)}
                >
                  Continue to Financials &rarr;
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Financials & target location */}
        {activeStep === 1 && (
          <div className="form-step-panel fade-in">
            <h2>Financial Planning & Target Market</h2>
            <p className="subtitle">Configure your operational parameters in localized currency symbols. Values are translated dynamically using relative PPP.</p>

            <div className="input-split-layout">
              <div className="layout-col">
                <div className="form-group">
                  <label>Founder's Startup Experience</label>
                  <select value={founderExperience} onChange={(e) => setFounderExperience(e.target.value)}>
                    <option value="Novice">Novice (First business)</option>
                    <option value="Intermediate">Intermediate (Prior management background)</option>
                    <option value="Experienced">Experienced (Serial entrepreneur)</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Target Location</label>
                  <select value={countryCode} onChange={(e) => handleCountryChange(e.target.value)}>
                    {countries.length > 0 ? (
                      countries.map(c => (
                        <option key={c.code} value={c.code}>{c.name} ({c.currency})</option>
                      ))
                    ) : (
                      <option value="IN">India (INR)</option>
                    )}
                  </select>
                  <span className="info-helper">Exchange indicators are resolved against US baseline values.</span>
                </div>

                <div className="form-group">
                  <label>Starting Capital ({activeCountry.currency} {activeCountry.symbol})</label>
                  <input 
                    type="number" 
                    value={startingCapital}
                    onChange={(e) => setStartingCapital(e.target.value)}
                    min="10" 
                  />
                  <span className="info-helper">Suggested minimum baseline: {activeCountry.symbol}{local_params_suggested(industry, activeCountry.ppp).toLocaleString()}</span>
                </div>
              </div>

              <div className="layout-col">
                <div className="form-group">
                  <label>Monthly Marketing Budget ({activeCountry.symbol})</label>
                  <input 
                    type="number" 
                    value={marketingBudget}
                    onChange={(e) => setMarketingBudget(e.target.value)}
                    min="0" 
                  />
                </div>

                <div className="form-group">
                  <label>Expected Team Size (Employees)</label>
                  <input 
                    type="number" 
                    value={teamSize}
                    onChange={(e) => setTeamSize(e.target.value)}
                    min="1" 
                    max="100"
                  />
                </div>

                <div className="form-group">
                  <label>Average Transaction / Subscription Value ({activeCountry.symbol})</label>
                  <input 
                    type="number" 
                    value={avgTxValue}
                    onChange={(e) => setAvgTxValue(e.target.value)}
                    min="0.1" 
                    step="any"
                  />
                </div>
              </div>
            </div>

            <div className="button-footer">
              <button 
                type="button" 
                className="btn btn-secondary" 
                onClick={() => setActiveStep(0)}
              >
                &larr; Back
              </button>
              <button 
                type="button" 
                className="btn btn-primary" 
                onClick={handlePredict}
              >
                Predict Business Viability 🚀
              </button>
            </div>
          </div>
        )}

        {/* Step 3: ML Results Dashboard */}
        {activeStep === 2 && prediction && (
          <div className="report-panel fade-in">
            <div className="report-header">
              <div>
                <h2>Venture Feasibility Analysis</h2>
                <p className="subtitle" style={{ marginBottom: 0 }}>Outcome vector computed by Random Forest Classifier.</p>
              </div>
              <button className="btn btn-secondary" onClick={() => setActiveStep(1)}>
                &larr; Adjust Parameters
              </button>
            </div>

            {/* Main KPI Card Grid */}
            <div className="kpi-grid">
              {/* Survival rate Card */}
              {(() => {
                const status = getSurvivalStatus(prediction.survival_probability);
                return (
                  <div className="kpi-card" style={{ backgroundColor: status.bg, borderColor: status.border }}>
                    <span className="kpi-label" style={{ color: '#94a3b8' }}>ML PREDICTED SURVIVAL PROBABILITY (3 YEAR)</span>
                    <span className="kpi-value" style={{ color: status.pctColor }}>{prediction.survival_probability}%</span>
                    <span className="kpi-tag" style={{ backgroundColor: status.color, color: '#0b0f19' }}>
                      {status.label}
                    </span>
                  </div>
                );
              })()}

              {/* Profit Card */}
              <div className="kpi-card" style={{ 
                backgroundColor: prediction.expected_annual_profit >= 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                borderColor: prediction.expected_annual_profit >= 0 ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'
              }}>
                <span className="kpi-label" style={{ color: '#94a3b8' }}>ML EXPECTED ANNUAL NET INCOME</span>
                <span className="kpi-value" style={{ 
                  color: prediction.expected_annual_profit >= 0 ? '#34d399' : '#f87171'
                }}>
                  {prediction.expected_annual_profit >= 0 ? '+' : ''}
                  {activeCountry.symbol}{prediction.expected_annual_profit.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                </span>
                <span className="kpi-tag" style={{ 
                  backgroundColor: prediction.expected_annual_profit >= 0 ? '#10b981' : '#ef4444',
                  color: '#0b0f19'
                }}>
                  Currency: {activeCountry.currency}
                </span>
              </div>
            </div>

            <div className="report-details-split">
              {/* Expense Breakdown Bar */}
              <div className="details-col">
                <h3>Annualized Expense Allocation</h3>
                <p className="subtitle">Estimated operating burn rates adjusted for local commercial variables:</p>
                
                <div className="cost-breakdown-list">
                  <div className="cost-item">
                    <span className="cost-name">Rent (Annual Commercial Estimate)</span>
                    <span className="cost-amount">{activeCountry.symbol}{prediction.breakdown.rent_annual.toLocaleString()}</span>
                  </div>
                  <div className="cost-item">
                    <span className="cost-name">Staff & Salaries (Annual Team Estimate)</span>
                    <span className="cost-amount">{activeCountry.symbol}{prediction.breakdown.labor_annual.toLocaleString()}</span>
                  </div>
                  <div className="cost-item">
                    <span className="cost-name">Marketing (Annual Outlay)</span>
                    <span className="cost-amount">{activeCountry.symbol}{prediction.breakdown.marketing_annual.toLocaleString()}</span>
                  </div>
                  <div className="cost-total-line"></div>
                  <div className="cost-item total">
                    <span>Total Fixed Costs / Burn Rate (Annual)</span>
                    <span>{activeCountry.symbol}{(prediction.breakdown.rent_annual + prediction.breakdown.labor_annual + prediction.breakdown.marketing_annual).toLocaleString()}</span>
                  </div>
                </div>

                <div className="cost-progress-bar">
                  {(() => {
                    const total = prediction.breakdown.rent_annual + prediction.breakdown.labor_annual + prediction.breakdown.marketing_annual;
                    if (total === 0) return null;
                    const rentPct = (prediction.breakdown.rent_annual / total) * 100;
                    const laborPct = (prediction.breakdown.labor_annual / total) * 100;
                    const mktgPct = (prediction.breakdown.marketing_annual / total) * 100;
                    return (
                      <div className="stacked-progress">
                        <div className="progress-slice rent" style={{ width: `${rentPct}%` }} title="Rent"></div>
                        <div className="progress-slice labor" style={{ width: `${laborPct}%` }} title="Labor"></div>
                        <div className="progress-slice marketing" style={{ width: `${mktgPct}%` }} title="Marketing"></div>
                      </div>
                    );
                  })()}
                  <div className="progress-legend">
                    <span className="legend-item"><span className="dot rent"></span> Rent</span>
                    <span className="legend-item"><span className="dot labor"></span> Labor</span>
                    <span className="legend-item"><span className="dot marketing"></span> Marketing</span>
                  </div>
                </div>
              </div>

              {/* Diagnostic Explanations */}
              <div className="details-col">
                <h3>💡 ML Model Diagnostics</h3>
                <div className="diagnostics-box">
                  {prediction.diagnostics.length > 0 ? (
                    prediction.diagnostics.map((diag, i) => (
                      <div key={i} className={`diagnostic-card ${prediction.survival_probability < 50.0 ? 'warning' : 'success'}`}>
                        <span className="diag-icon">{prediction.survival_probability < 50.0 ? '🚨' : '✨'}</span>
                        <p>{diag}</p>
                      </div>
                    ))
                  ) : (
                    <div className="diagnostic-card success">
                      <span className="diag-icon">✨</span>
                      <p>Venture parameters correspond to standard viable metrics in this target economy.</p>
                    </div>
                  )}
                </div>
                <p className="disclaimer">*Diagnostics mapped from feature weights in local random forest models.*</p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// Inline helper for Suggested Capital baseline estimation
function local_params_suggested(ind, ppp) {
  const base_caps = {
    "SaaS / Mobile Apps": 15000,
    "Bakery / Cafe / Food": 80000,
    "Boutique Retail / E-commerce": 50000,
    "Local Services (Salon, Gym, Repairs)": 25000,
    "E-Learning / Online Academy": 8000,
    "Agrotech / Vertical Farming": 120000,
    "Real Estate / Property Management": 60000,
    "Healthcare Clinic / Wellness Care": 100000,
    "Logistics / Delivery Service": 90000
  };
  return Math.round((base_caps[ind] || 40000) * ppp);
}
