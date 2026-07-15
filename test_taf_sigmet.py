import sys
import json
from main import AviationWeatherParser, AviationWeatherML, LocalRAGEngine, DecisionAgent

def test_taf_sigmet_parsing():
    print("--- Running TAF & SIGMET Parsing Verification ---")
    parser = AviationWeatherParser()

    # 1. Test TAF Parsing (DFW TAF forecasting wind gusts and temporary thunderstorms)
    taf = "TAF KDFW 152340Z 1600/1706 18015G25KT 5SM -SHRA OVC020 TEMPO 1602/1606 1 1/2SM TSRA OVC008"
    f_taf = parser.parse(taf, "US")
    print(f"KDFW TAF parsed features:\n{f_taf.model_dump()}")
    # TAF worst-case sustained: 15, gust: 25, visibility: 1.5, ceiling: 800 ft
    assert f_taf.wind_speed == 15
    assert f_taf.wind_gust == 25
    assert f_taf.visibility == 1.5
    assert f_taf.ceiling == 800
    assert f_taf.report_type == "TAF"
    print("TAF test passed!")

    # 2. Test SIGMET Parsing (Turbulence SIGMET PAPA 3)
    sigmet = "SIGMET PAPA 3 VALID 150000/150400 KKCI - SEV TURB FCST BETWEEN FL280 AND FL370"
    f_sig = parser.parse(sigmet, "US")
    print(f"PAPA 3 SIGMET parsed features:\n{f_sig.model_dump()}")
    assert f_sig.severe_turbulence_alert is True
    assert f_sig.report_type == "SIGMET"
    print("SIGMET test passed!")

def test_ml_predictions():
    print("--- Running TAF/SIGMET ML Prediction Verification ---")
    ml = AviationWeatherML()
    
    # Severe TAF prediction
    taf = "TAF KDFW 152340Z 1600/1706 18015G25KT 5SM -SHRA OVC020 TEMPO 1602/1606 1 1/2SM TSRA OVC008"
    res_taf = ml.predict(taf, "US")
    print("Severe TAF predicted risk:", res_taf["risk"])
    assert res_taf["risk"] in ["High", "Medium"]
    
    # Severe SIGMET prediction
    sigmet = "SIGMET PAPA 3 VALID 150000/150400 KKCI - SEV TURB FCST BETWEEN FL280 AND FL370"
    res_sig = ml.predict(sigmet, "US")
    print("Severe SIGMET predicted risk:", res_sig["risk"])
    assert res_sig["risk"] == "High"
    print("ML predictions test passed!")

def test_rag_retrieval():
    print("--- Running RAG TAF/SIGMET Retrieval Verification ---")
    rag = LocalRAGEngine()
    
    # Check if database has our new manual
    results = rag.retrieve("What does TEMPO mean in a TAF?")
    print("RAG query 'What does TEMPO mean in a TAF?':\n", results)
    assert "TEMPO" in results or "taf_sigmet_manual.md" in results
    print("RAG retrieval test passed!")

if __name__ == "__main__":
    try:
        test_taf_sigmet_parsing()
        test_ml_predictions()
        test_rag_retrieval()
        print("\nALL TAF AND SIGMET TEST SUITE PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nTEST SUITE FAILED: {e}")
        sys.exit(1)
