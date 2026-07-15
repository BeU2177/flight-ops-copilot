import sys
from main import LocalRAGEngine

def test_runway_headings():
    print("--- Running Airport Runways Verification ---")
    rag = LocalRAGEngine()
    print("Ingesting datasets...")
    rag.ingest()

    # 1. Verify VOMM (Chennai International Airport - VO Region)
    vomm_info = rag.retrieve("What are the runway headings for VOMM?")
    print("VOMM Retrieval Results:\n", vomm_info)
    assert "VOMM" in vomm_info
    assert "Runways:" in vomm_info
    assert "07/25" in vomm_info or "12/30" in vomm_info
    print("VOMM runway verification passed!")

    # 2. Verify VOBL (Kempegowda International Airport - VO Region)
    vobl_info = rag.retrieve("What are the runway headings for VOBL?")
    print("VOBL Retrieval Results:\n", vobl_info)
    assert "VOBL" in vobl_info
    assert "09L/27R" in vobl_info or "09R/27L" in vobl_info
    print("VOBL runway verification passed!")

    # 3. Verify KDEN (Denver International Airport - K Region)
    kden_info = rag.retrieve("What are the runway headings for KDEN?")
    print("KDEN Retrieval Results:\n", kden_info)
    assert "KDEN" in kden_info
    assert "16L/34R" in kden_info or "17L/35R" in kden_info
    print("KDEN runway verification passed!")

if __name__ == "__main__":
    try:
        test_runway_headings()
        print("\nALL RUNWAY HEADINGS VERIFICATION TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nTEST SUITE FAILED: {e}")
        sys.exit(1)
