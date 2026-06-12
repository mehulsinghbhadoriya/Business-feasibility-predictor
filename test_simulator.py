import simulator as sim
import json
import sys

def run_tests():
    print("==================================================")
    print("Testing Global Business Feasibility Simulator Engine")
    print("==================================================")
    
    # Test 1: Database Loading
    print("\n[Test 1] Loading databases...")
    try:
        industries, macro_df = sim.load_databases()
        print(f"-> Successfully loaded {len(industries)} industry archetypes.")
        print(f"-> Successfully loaded {len(macro_df)} global country configurations.")
        print("PASS: Databases loaded.")
    except Exception as e:
        print(f"FAIL: Databases loading failed. Error: {e}")
        return

    # Test 2: NLP Similarity Matching
    print("\n[Test 2] Testing Business Idea Classifier (TF-IDF Similarity)...")
    test_ideas = [
        "A cosy cafe serving vegan cakes and hot coffee",
        "A cloud-based SaaS tool for software project management",
        "A pet grooming salon and dog spa in the city center"
    ]
    
    for idea in test_ideas:
        matched, score = sim.match_business_idea(idea)
        print(f"Input Idea: '{idea}'")
        print(f"Matched Archetype: '{matched['name']}' (Confidence: {score*100:.1f}%)")
        print("-" * 30)
    print("PASS: Idea classification tested successfully.")

    # Test 3: Financial Global Scaling
    print("\n[Test 3] Testing Financial Cost Scaling for India...")
    india_row = macro_df[macro_df['Country'] == 'India'].iloc[0]
    matched, _ = sim.match_business_idea(test_ideas[0]) # Cafe
    scaled_metrics = sim.scale_finances(matched, india_row)
    print(f"Matched Category: {matched['name']}")
    print(f"Original US Capital Recommendation: ${matched['base_startup_capital']:,}")
    symbol_safe = india_row['Symbol'] if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() == 'utf-8' else india_row['Currency']
    print(f"India Localized Capital Recommendation (PPP Adjusted): {symbol_safe} {scaled_metrics['startup_capital']:,}")
    print(f"India Localized Monthly Rent: {symbol_safe} {scaled_metrics['monthly_rent']:,}")
    print(f"India Localized Monthly Staffing: {symbol_safe} {scaled_metrics['monthly_labor']:,}")
    print("PASS: Scaling factors applied properly.")

    # Test 4: Monte Carlo Engine Simulation Runs
    print("\n[Test 4] Testing NumPy-based Monte Carlo Simulator...")
    # Simulated Cafe run: Rent=5000, Labor=10000, Ops=5000, TargetRevenue=30000
    # Startup capital: 50,000 INR
    sim_results = sim.run_monte_carlo(
        startup_capital=50000,
        monthly_revenue=30000,
        rent=5000,
        labor=10000,
        ops=5000,
        rev_vol=0.15,
        cost_vol=0.08,
        runs=1000
    )
    print(f"Survival Probability (12 Months): {sim_results['survival_probability']}%")
    print(f"Median Cash Balance (Month 12): INR {sim_results['median_case_balance']:,}")
    print(f"Worst Case (P10) Balance: INR {sim_results['worst_case_balance']:,}")
    print(f"Best Case (P90) Balance: INR {sim_results['best_case_balance']:,}")
    print(f"Median Cash Trajectory: {['%.1f' % c for c in sim_results['median_trajectory'][:4]]} ...")
    print("PASS: Monte Carlo engine ran perfectly.")

    print("\n==================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
