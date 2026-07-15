import os
import streamlit as st
import logging

# Ensure dotenv is loaded
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import main

# Title and Layout
st.set_page_config(
    page_title="Flight Operations CoPilot",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic CSS Styling
st.markdown("""
<style>
    /* Styling settings for dark mode glassmorphism look */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1a1c29 0%, #0d0e15 100%);
        color: #e2e8f0;
    }
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }
    .agent-header {
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("<div class='main-title'>✈️ AI Flight Operations CoPilot</div>", unsafe_allow_html=True)

# Initialize Session State Agent
@st.cache_resource
def get_ops_agent():
    return main.OpsCopilotAgent()

with st.spinner("Initializing Flight Operations System... (Index Verification in progress)"):
    agent = get_ops_agent()

# Sidebar config
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # Show active LLM client type
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    if has_groq:
        st.success("🟢 LLM Client: Groq Cloud API")
        st.caption(f"Model: {os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant')}")
    else:
        st.warning("🔵 LLM Client: Local Ollama")
        st.caption("Model: qwen3:8b")
        
    st.markdown("---")
    st.markdown("### 💡 Try these Queries:")
    st.code("KJFK 142351Z 34018G30KT 1/2SM R04R/2000FT -SN BLSN FZFG OVC008 M08/M09 A2982")
    st.code("What are the active runway headings for airport VOMM?")
    st.code("Calculate landing distance for a B738 on dry runway at sea level, 65,000 kg.")

# User Input
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
query = st.text_input("Enter your flight operations query or METAR code:", placeholder="e.g. KDEN 142253Z 02022G35KT 3/4SM -SN BLSN FZFG")
submit = st.button("Analyze & Dispatch", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

if submit and query:
    with st.spinner("Processing Agent Routing..."):
        try:
            # Classify Intent
            intent = agent.classify_intent(query)
            
            # Map Agent Name
            agent_map = {
                "WEATHER": "WeatherAgent (Meteorological Analyst)",
                "RAG": "RAGAgent (Operations Manual Retriever)",
                "CALCULATOR": "PerformanceAgent (Flight Calculator)",
                "DECISION": "DecisionAgent (Command Orchestrator)"
            }
            routed_agent = agent_map.get(intent, "DecisionAgent (Command Orchestrator)")
            
            # Columns for Routing Metadata
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Routed Specialist", routed_agent)
            with col2:
                st.metric("Classified Intent", intent)
                
            # Execute Agent Pipeline
            response = agent.run(query)
            
            # Output Display Card
            st.markdown("### 📋 Copilot Resolution Output")
            st.markdown(f"<div class='glass-card'>{response}</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Execution Error: {e}")
