import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import main

def test_intent_routing():
    print("--- Running Bug 6: Intent Routing Test ---")
    agent = main.OpsCopilotAgent()
    
    # Query with weight-related keywords
    q1 = "Check weight limits: TOW 83500 kg, MTOW 79000 kg"
    intent1 = agent.classify_intent(q1)
    print(f"Query: '{q1}'")
    print(f"Classified Intent: {intent1}")
    assert intent1 == "CALCULATOR", f"Expected CALCULATOR, got {intent1}"
    print("Bug 6 test passed!")

def test_weight_verification():
    print("--- Running Bug 4: Weight Verification Test ---")
    calc = main.FlightCalculator()
    
    # Scenario: Overweight (83,500 kg TOW vs 79,000 kg MTOW)
    res = calc.verify_weight_limits(zfw=58500.0, fuel=16000.0, payload=9000.0, mtow=79000.0)
    print(f"TOW: {res['takeoff_weight_kg']} kg, MTOW: {res['max_takeoff_weight_kg']} kg")
    print(f"Is Legal Takeoff: {res['is_legal_takeoff']}")
    print(f"Warning: {res['weight_warning']}")
    
    assert res['takeoff_weight_kg'] == 83500.0
    assert res['is_legal_takeoff'] is False
    assert "CRITICAL WEIGHT VIOLATION" in res['weight_warning']
    print("Bug 4 test passed!")

if __name__ == "__main__":
    try:
        test_intent_routing()
        print()
        test_weight_verification()
        print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
