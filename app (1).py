"""
CO₂ Reduction AI Agent - HuggingFace Version (Streamlit Cloud Optimized)
"""

import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import re

# LangChain imports (wrapped in try-except for safety)
try:
    from langchain_huggingface import HuggingFaceEndpoint
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("LangChain not available - using smart response engine only")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="CO₂ Reduction Platform | Environmental AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== SESSION STATE ====================
if 'total_savings' not in st.session_state:
    st.session_state.total_savings = 0
if 'queries_count' not in st.session_state:
    st.session_state.queries_count = 0
if 'history' not in st.session_state:
    st.session_state.history = []
if 'user_query' not in st.session_state:
    st.session_state.user_query = ''
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'trigger_submit' not in st.session_state:
    st.session_state.trigger_submit = False

# HuggingFace LLM settings
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = "huggingface"
if 'model_name' not in st.session_state:
    st.session_state.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
if 'use_llm' not in st.session_state:
    st.session_state.use_llm = True

# ==================== PROFESSIONAL CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main body background - subtle gradient instead of white */
    .main {
        background: linear-gradient(180deg, #f0fdf4 100%, #e0f2fe 50%, #fef3c7 0%);
    }
    .stApp {
        background: linear-gradient(180deg, #f0fdf4 100%, #e0f2fe 50%, #fef3c7 0%);
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    @keyframes pulse { 0%,100%{opacity:1;}50%{opacity:.7;} }
    
    /* Animated Icons */
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    @keyframes wiggle {
        0%, 100% { transform: rotate(0deg); }
        25% { transform: rotate(-15deg); }
        75% { transform: rotate(15deg); }
    }
    .icon-animated {
        display: inline-block;
        animation: bounce 2s ease-in-out infinite;
    }
    .icon-animated:hover {
        animation: wiggle 0.5s ease-in-out;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    [data-testid="stSidebar"] { display: none; }
    
    .hero-section {
        position: relative;
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.35) 0%, rgba(16, 185, 129, 0.35) 50%, rgba(5, 150, 105, 0.35) 100%),
                    url('https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=1600&q=80') center/cover;
        padding: 5rem 2rem;
        border-radius: 0;
        margin: -6rem -6rem 3rem -6rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        overflow: hidden;
        filter: brightness(0.8);
    }
    
    .hero-section::before {
        content: '';
        position: absolute; top: 10%; left: -10%;
        width: 300px; height: 300px;
        background: url('https://cdn-icons-png.flaticon.com/512/2913/2913133.png') no-repeat center;
        background-size: contain; opacity: 0.2;
        animation: slideFloat1 20s linear infinite; z-index: 1;
    }
    .hero-section::after {
        content: '';
        position: absolute; top: 60%; right: -10%;
        width: 250px; height: 250px;
        background: url('https://cdn-icons-png.flaticon.com/512/3649/3649180.png') no-repeat center;
        background-size: contain; opacity: 0.2;
        animation: slideFloat2 25s linear infinite; z-index: 1;
    }
    @keyframes slideFloat1 {
        0% { transform: translateX(-100px) translateY(0) rotate(0deg); left: -10%; }
        50% { transform: translateX(50vw) translateY(-30px) rotate(180deg); }
        100% { transform: translateX(100vw) translateY(0) rotate(360deg); left: 110%; }
    }
    @keyframes slideFloat2 {
        0% { transform: translateX(100px) translateY(0) rotate(0deg); right: -10%; }
        50% { transform: translateX(-50vw) translateY(30px) rotate(-180deg); }
        100% { transform: translateX(-100vw) translateY(0) rotate(-360deg); right: 110%; }
    }
    .hero-content { position: relative; z-index: 2; max-width: 1200px; margin: 0 auto; text-align: center; animation: fadeInUp 1s ease-out; }
    .hero-title { font-size: 4rem; font-weight: 800; color: white !important; margin-bottom: 1rem; text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5); line-height: 1.2; }
    .hero-subtitle { font-size: 1.5rem; color: white !important; margin-bottom: 2rem; font-weight: 400; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); }
    .hero-stats { display: flex; justify-content: center; gap: 3rem; margin-top: 2rem; flex-wrap: wrap; }
    .hero-stat-item { background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(10px); padding: 1.5rem 2.5rem; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.3); animation: fadeInUp 1.2s ease-out; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); }
    .hero-stat-value { font-size: 2.5rem; font-weight: 700; color: white; display: block; }
    .hero-stat-label { font-size: 0.9rem; color: rgba(255, 255, 255, 0.95); display: block; margin-top: 0.5rem; }

    .glass-card { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 24px; padding: 2.5rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.8); transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); animation: fadeInUp 0.8s ease-out; margin-bottom: 2rem; }
    .glass-card:hover { transform: translateY(-8px); box-shadow: 0 20px 60px rgba(16, 185, 129, 0.2); border-color: rgba(16, 185, 129, 0.3); }

    .chat-message { padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem; animation: slideIn 0.3s ease-out; }
    .user-message { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; margin-left: 20%; }
    .assistant-message { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); color: #1f2937; margin-right: 20%; border: 2px solid #86efac; }

    .metric-card-pro { background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%); padding: 2rem; border-radius: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08); border: 2px solid #e8f8f5; transition: all 0.3s ease; height: 100%; position: relative; overflow: hidden; }
    .metric-card-pro::before { content: ''; position: absolute; top: -50%; right: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(16, 185, 129, 0.1) 0%, transparent 70%); animation: float 6s ease-in-out infinite; }
    .metric-card-pro:hover { transform: scale(1.05) rotate(-1deg); box-shadow: 0 12px 40px rgba(16, 185, 129, 0.2); border-color: #10b981; }
    .metric-icon { font-size: 3rem; margin-bottom: 1rem; display: block; animation: float 3s ease-in-out infinite; }
    .metric-value-pro { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #10b981 0%, #059669 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 1rem 0; position: relative; }
    .metric-label-pro { font-size: 0.95rem; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }

    .input-section { background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%); padding: 3rem; border-radius: 24px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08); margin-bottom: 2rem; }
    .section-title { font-size: 2rem; font-weight: 700; color: #1f2937; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem; }
    .stTextArea textarea { border-radius: 16px; border: 2px solid #e5e7eb; padding: 1.5rem; font-size: 1.1rem; transition: all 0.3s ease; }
    .stTextArea textarea:focus { border-color: #10b981; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1); }

    .stButton > button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; border: none; padding: 1.2rem 3rem; font-size: 1.1rem; font-weight: 600; border-radius: 50px; box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3); transition: all 0.3s ease; text-transform: uppercase; letter-spacing: 1px; }
    .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 12px 35px rgba(16, 185, 129, 0.4); background: linear-gradient(135deg, #059669 0%, #047857 100%); }

    .example-card { background: white; padding: 1.5rem; border-radius: 16px; border: 2px solid #e5e7eb; transition: all 0.3s ease; cursor: pointer; text-align: left; height: 100%; }
    .example-card:hover { border-color: #10b981; transform: translateY(-4px); box-shadow: 0 8px 25px rgba(16, 185, 129, 0.15); }
    .example-icon { font-size: 2rem; margin-bottom: 0.5rem; display: block; }

    .results-section { background: white; padding: 3rem; border-radius: 24px; box-shadow: 0 10px 50px rgba(0, 0, 0, 0.1); border: 2px solid #e8f8f5; animation: fadeInUp 0.6s ease-out; }

    .impact-banner { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 2rem; border-radius: 20px; text-align: center; margin: 2rem 0; box-shadow: 0 8px 30px rgba(16, 185, 129, 0.3); }
    .impact-banner h2 { color: white; margin: 0; font-size: 2rem; }

    .llm-badge { display: inline-block; background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-left: 1rem; }

    .stSpinner > div { border-top-color: #10b981 !important; }

    .tech-icon { font-size: 2.5rem; display: inline-block; margin: 0 0.5rem; }
    .tech-item { display: flex; align-items: center; gap: 1rem; padding: 0.8rem 0; font-size: 1.1rem; }

    .additional-resources-section {
        position: relative;
        background:
            linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)),
            url("https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?w=1600&q=80") center/cover;
        padding: 4rem 2rem;
        margin: 4rem -6rem 0 -6rem;
        border-radius: 0;
    }

    .copyright-footer { background: linear-gradient(135deg, #0f766e 0%, #0d9488 50%, #14b8a6 100%); padding: 2rem 0; text-align: center; margin: 4rem -6rem 0 -6rem; border-radius: 0; }
    .copyright-text { color: rgba(255, 255, 255, 0.95); font-size: 1rem; margin: 0; font-weight: 400; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

# ==================== DATA ====================

CO2_DATA = [
    {"Activity": "Car (Petrol, 20 km)", "Avg_CO2_Emission(kg/day)": 4.6, "Category": "Transport", "Icon": "🚗"},
    {"Activity": "Bus (20 km)", "Avg_CO2_Emission(kg/day)": 1.2, "Category": "Transport", "Icon": "🚌"},
    {"Activity": "Bicycle (20 km)", "Avg_CO2_Emission(kg/day)": 0.0, "Category": "Transport", "Icon": "🚴"},
    {"Activity": "AC usage (8 hrs/day)", "Avg_CO2_Emission(kg/day)": 6.0, "Category": "Household", "Icon": "❄️"},
    {"Activity": "LED Bulb (5 hrs/day)", "Avg_CO2_Emission(kg/day)": 0.05, "Category": "Household", "Icon": "💡"},
    {"Activity": "Old Bulb (5 hrs/day)", "Avg_CO2_Emission(kg/day)": 0.2, "Category": "Household", "Icon": "🔆"},
    {"Activity": "Meat-based diet", "Avg_CO2_Emission(kg/day)": 7.0, "Category": "Food", "Icon": "🍖"},
    {"Activity": "Vegetarian diet", "Avg_CO2_Emission(kg/day)": 2.0, "Category": "Food", "Icon": "🥗"},
    {"Activity": "Online shopping (1)", "Avg_CO2_Emission(kg/day)": 1.0, "Category": "Lifestyle", "Icon": "📦"},
    {"Activity": "Local shopping (1)", "Avg_CO2_Emission(kg/day)": 0.3, "Category": "Lifestyle", "Icon": "🛍️"},
]

SUSTAINABILITY_TIPS = [
    "For transport, switching from petrol cars to public buses can reduce CO2 emissions by up to 74%. Buses emit approximately 1.2 kg CO2 per 20 km compared to 4.6 kg for petrol cars.",
    "Cycling is the most eco-friendly transport option with zero CO2 emissions. It's ideal for distances under 10 km and also provides health benefits.",
    "Carpooling reduces individual carbon footprint significantly. Sharing a ride with 3 colleagues reduces per-person emissions from 4.6 kg to about 1.5 kg CO2 per day.",
    "Electric vehicles (EVs) and hybrid models can reduce transport emissions by 60-80% compared to traditional petrol vehicles.",
    "LED bulbs use 75% less energy than traditional incandescent bulbs. Switching from old bulbs to LED can reduce emissions from 0.2 kg to 0.05 kg CO2 per day.",
    "Air conditioning is a major household energy consumer. Using AC efficiently (setting temperature to 24°C, regular maintenance) can reduce emissions from 6.0 kg to 4.0 kg CO2 per day.",
    "Plant-based diets have significantly lower carbon footprints. Vegetarian diets emit about 2.0 kg CO2 per day compared to 7.0 kg for meat-based diets - a 71% reduction.",
    "Local shopping reduces packaging and transportation emissions. It emits 0.3 kg CO2 compared to 1.0 kg for online shopping due to reduced delivery logistics."
]

# ==================== IMPROVED QUERY PARSING ====================

def parse_query_category(query: str) -> str:
    """Identify which category the query is about"""
    query_lower = query.lower()
    
    food_keywords = ['food', 'diet', 'meat', 'vegetarian', 'vegan', 'eat', 'eating', 'meal', 'plant-based', 'consume', 'consumption']
    lifestyle_keywords = ['shop', 'shopping', 'online', 'local', 'purchase', 'buy', 'buying']
    household_keywords = ['electricity', 'bulb', 'light', 'led', 'household', 'home', 'energy', 'power', 'appliance']
    transport_keywords = [' car ', ' drive', 'driving', 'petrol', ' bus ', 'bicycle', ' bike ', 'transport', 'commute', 'travel', 'vehicle', 'car,', 'car.', 'car?']
    ac_keywords = [' ac ', 'air condition', 'air-condition', 'a/c', 'ac usage', 'ac use']
    
    if any(kw in query_lower for kw in food_keywords):
        return "Food"
    elif any(kw in query_lower for kw in lifestyle_keywords):
        return "Lifestyle"
    elif any(kw in query_lower for kw in ac_keywords):
        return "Household"
    elif any(kw in query_lower for kw in household_keywords):
        return "Household"
    elif any(kw in query_lower for kw in transport_keywords):
        return "Transport"
    
    return "General"

def find_activity_from_query(query: str, CO2_DATA: list) -> dict:
    """Enhanced activity finder with better matching"""
    query_lower = query.lower()
    category = parse_query_category(query)
    
    if category == "Food":
        if 'meat' in query_lower:
            for item in CO2_DATA:
                if 'Meat' in item['Activity']:
                    return item
        if 'vegetarian' in query_lower or 'vegan' in query_lower or 'plant' in query_lower:
            for item in CO2_DATA:
                if 'Vegetarian' in item['Activity']:
                    return item
        food_items = [item for item in CO2_DATA if item['Category'] == 'Food']
        if food_items:
            return max(food_items, key=lambda x: x['Avg_CO2_Emission(kg/day)'])
    
    if category == "Lifestyle":
        if 'online' in query_lower:
            for item in CO2_DATA:
                if 'Online' in item['Activity']:
                    return item
        if 'local' in query_lower:
            for item in CO2_DATA:
                if 'Local' in item['Activity']:
                    return item
        lifestyle_items = [item for item in CO2_DATA if item['Category'] == 'Lifestyle']
        if lifestyle_items:
            return max(lifestyle_items, key=lambda x: x['Avg_CO2_Emission(kg/day)'])
    
    if category == "Household":
        if 'ac' in query_lower or 'air condition' in query_lower:
            for item in CO2_DATA:
                if 'AC' in item['Activity']:
                    return item
        if 'led' in query_lower:
            for item in CO2_DATA:
                if 'LED' in item['Activity']:
                    return item
        if 'bulb' in query_lower and ('old' in query_lower or 'traditional' in query_lower):
            for item in CO2_DATA:
                if 'Old Bulb' in item['Activity']:
                    return item
        household_items = [item for item in CO2_DATA if item['Category'] == 'Household']
        if household_items:
            return max(household_items, key=lambda x: x['Avg_CO2_Emission(kg/day)'])
    
    if category == "Transport":
        if 'car' in query_lower or 'petrol' in query_lower or 'drive' in query_lower or 'driving' in query_lower:
            for item in CO2_DATA:
                if 'Car' in item['Activity']:
                    return item
        if 'bus' in query_lower:
            for item in CO2_DATA:
                if 'Bus' in item['Activity']:
                    return item
        if 'bicycle' in query_lower or 'bike' in query_lower or 'cycle' in query_lower:
            for item in CO2_DATA:
                if 'Bicycle' in item['Activity']:
                    return item
        transport_items = [item for item in CO2_DATA if item['Category'] == 'Transport']
        if transport_items:
            return max(transport_items, key=lambda x: x['Avg_CO2_Emission(kg/day)'])
    
    return None

# ==================== INTELLIGENT RESPONSE GENERATOR ====================

def generate_smart_response(query: str, CO2_DATA: list, relevant_tips: list) -> tuple:
    """Generate intelligent response without relying on LLM agent"""
    
    category = parse_query_category(query)
    current_activity = find_activity_from_query(query, CO2_DATA)
    
    category_icons = {
        "Transport": '<span class="icon-animated">🚗</span>',
        "Household": '<span class="icon-animated">🏠</span>',
        "Food": '<span class="icon-animated">🥗</span>',
        "Lifestyle": '<span class="icon-animated">🛍️</span>',
        "General": '<span class="icon-animated">🌍</span>'
    }
    
    if not current_activity:
        response = f"""**{category_icons.get(category, category_icons['General'])} Sustainability Guidance**

Based on your query about {category.lower() if category != "General" else "sustainability"}, here are key recommendations:

"""
        if relevant_tips:
            for idx, tip in enumerate(relevant_tips[:3], 1):
                response += f"{idx}. {tip}\n\n"
        else:
            response += "• Focus on reducing emissions in transport, household energy, and food choices\n"
            response += "• Small daily changes can lead to significant annual CO₂ reductions\n"
            response += "• Consider alternatives that align with your lifestyle and location\n"
        
        return response, None, []
    
    category_items = [item for item in CO2_DATA if item['Category'] == current_activity['Category']]
    alternatives = [item for item in category_items if item != current_activity]
    alternatives.sort(key=lambda x: x['Avg_CO2_Emission(kg/day)'])
    
    response = f"""**<span class="icon-animated">{current_activity['Icon']}</span> Your Current Activity Analysis**

**Current Activity:** {current_activity['Activity']} {current_activity['Icon']}
**Category:** {current_activity['Category']}
**Daily CO₂ Emissions:** {current_activity['Avg_CO2_Emission(kg/day)']} kg

---

**<span class="icon-animated">💡</span> Recommended Alternatives:**

"""
    
    for idx, alt in enumerate(alternatives[:3], 1):
        emission_diff = current_activity['Avg_CO2_Emission(kg/day)'] - alt['Avg_CO2_Emission(kg/day)']
        if emission_diff > 0:
            reduction_pct = (emission_diff / current_activity['Avg_CO2_Emission(kg/day)']) * 100
            annual_savings = emission_diff * 365
            trees = int(annual_savings / 21)
            response += f"{idx}. **{alt['Activity']}** {alt['Icon']}\n"
            response += f"   • Reduces to: {alt['Avg_CO2_Emission(kg/day)']} kg CO₂/day\n"
            response += f"   • Daily savings: {emission_diff:.2f} kg CO₂\n"
            response += f"   • Reduction: {reduction_pct:.0f}%\n"
            response += f"   • Annual impact: {annual_savings:.0f} kg CO₂ (≈ {trees} trees planted)\n\n"
    
    if relevant_tips:
        response += "\n**<span class='icon-animated'>📚</span> Additional Insights:**\n\n"
        response += relevant_tips[0]
    
    return response, current_activity, alternatives

# ==================== HUGGINGFACE LLM CONFIGURATION ====================

def initialize_llm():
    """Initialize HuggingFace LLM"""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    try:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            # Try getting from Streamlit secrets
            try:
                hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
            except:
                pass
        
        if not hf_token:
            return None
        
        llm = HuggingFaceEndpoint(
            repo_id=st.session_state.model_name,
            huggingfacehub_api_token=hf_token,
            temperature=0.7,
            max_new_tokens=512
        )
        return llm
    except Exception as e:
        return None

# ==================== VECTOR STORE FUNCTIONS ====================

@st.cache_resource
def initialize_vector_store():
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Simple ChromaDB client (works on Streamlit Cloud)
        chroma_client = chromadb.Client()
        
        try:
            collection = chroma_client.create_collection(name="sustainability_tips")
        except:
            try:
                chroma_client.delete_collection(name="sustainability_tips")
            except:
                pass
            collection = chroma_client.create_collection(name="sustainability_tips")
        
        for idx, tip in enumerate(SUSTAINABILITY_TIPS):
            embedding = embedding_model.encode(tip, show_progress_bar=False).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[tip],
                ids=[f"tip_{idx}"]
            )
        
        return embedding_model, collection
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None, None

def retrieve_relevant_tips(query, embedding_model, collection, n_results=3):
    try:
        query_embedding = embedding_model.encode(query, show_progress_bar=False).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results['documents'][0] if results['documents'] else []
    except:
        return []

# ==================== CHART FUNCTIONS ====================

def create_comparison_chart(current_activity, alternatives):
    activities = [current_activity['Activity']] + [alt['Activity'] for alt in alternatives]
    emissions = [current_activity['Avg_CO2_Emission(kg/day)']] + [alt['Avg_CO2_Emission(kg/day)'] for alt in alternatives]
    colors = ['#ef4444'] + ['#10b981'] * len(alternatives)
    fig = go.Figure(data=[
        go.Bar(
            x=activities, y=emissions, marker_color=colors,
            marker_line_color='white', marker_line_width=2,
            text=[f"{e} kg" for e in emissions], textposition='outside',
            textfont=dict(size=14, color='#1f2937', weight='bold')
        )
    ])
    fig.update_layout(
        title=dict(text="CO₂ Emissions Comparison", font=dict(size=24, color='#1f2937', family='Inter')),
        xaxis_title="Activity", yaxis_title="CO₂ Emissions (kg/day)",
        xaxis=dict(showgrid=False, showline=True, linewidth=2, linecolor='#e5e7eb'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f3f4f6'),
        height=500, showlegend=False,
        plot_bgcolor='rgba(240, 253, 244, 0.3)', paper_bgcolor='white',
        font=dict(family='Inter', size=12), margin=dict(t=80, b=80, l=60, r=60)
    )
    return fig

def create_pie_chart(data):
    df = pd.DataFrame(data)
    category_emissions = df.groupby('Category')['Avg_CO2_Emission(kg/day)'].sum().reset_index()
    fig = px.pie(
        category_emissions, 
        values='Avg_CO2_Emission(kg/day)', 
        names='Category',
        color_discrete_sequence=['#10b981', '#3b82f6', '#ef4444', '#f59e0b'],
        hole=0.5
    )
    fig.update_traces(textposition='outside', textinfo='label+percent', marker=dict(line=dict(color='white', width=3)))
    fig.update_layout(height=400, showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1), font=dict(family='Inter', size=12), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# ==================== MAIN APP ====================

def main():
    # Hero Section
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="hero-title">🌍 CO₂ Reduction Platform</h1>
            <p class="hero-subtitle">
                AI-Powered Environmental Intelligence 
                <span class="llm-badge">🤖 Smart AI</span>
            </p>
            <div class="hero-stats">
                <div class="hero-stat-item">
                    <span class="hero-stat-value">{st.session_state.queries_count}</span>
                    <span class="hero-stat-label">Queries Analyzed</span>
                </div>
                <div class="hero-stat-item">
                    <span class="hero-stat-value">{st.session_state.total_savings:.1f}</span>
                    <span class="hero-stat-label">kg CO₂ Saved</span>
                </div>
                <div class="hero-stat-item">
                    <span class="hero-stat-value">{int(st.session_state.total_savings * 365 / 21)}</span>
                    <span class="hero-stat-label">Trees Equivalent</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize systems
    with st.spinner("🤖 Initializing AI Systems..."):
        embedding_model, collection = initialize_vector_store()
        llm = initialize_llm()
    
    if not embedding_model:
        st.error("❌ Failed to initialize embedding system")
        return
    
    # Main Content Area
    st.markdown("## <span class='icon-animated'>💬</span> Environmental AI Assistant", unsafe_allow_html=True)
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title"><span class="icon-animated">🔍</span> Ask Your Question</h3>', unsafe_allow_html=True)
        
        # Quick Examples
        st.markdown("#### <span class='icon-animated'>📝</span> Quick Examples", unsafe_allow_html=True)
        examples = [
            ("🚗", "Transport", "I drive 20 km daily using a petrol car. How can I reduce my CO₂ emissions?"),
            ("🏠", "Household", "How can I reduce CO₂ from household electricity usage?"),
            ("🥗", "Food", "What are eco-friendly food choices to reduce my carbon footprint?"),
            ("🛍️", "Lifestyle", "How does online shopping impact my carbon footprint?")
        ]
        cols = st.columns(2)
        for idx, (icon, category, example) in enumerate(examples):
            with cols[idx % 2]:
                if st.button(f"{icon} {category}", key=f"ex{idx}", use_container_width=True):
                    st.session_state.user_query = example
                    st.session_state.trigger_submit = True
                    st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        user_query = st.text_area(
            "Type your question here:",
            height=150,
            value=st.session_state.get('user_query', ''),
            placeholder="e.g., I use AC for 8 hours daily. How can I reduce emissions?",
            key="query_input"
        )
        
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            submit_button = st.button("🔍 Analyze & Get Recommendations", type="primary", use_container_width=True)
        with col_btn2:
            clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # FIXED: Removed chat_memory.clear() bug
        if clear_chat:
            st.session_state.conversation_history = []
            st.session_state.total_savings = 0
            st.session_state.queries_count = 0
            st.rerun()
    
    with col_right:
        st.markdown("### <span class='icon-animated'>🌍</span> Impact Preview", unsafe_allow_html=True)
        st.markdown("""
        <div class="image-card" style="height: 250px; margin-bottom: 1.5rem;">
            <img src="https://images.unsplash.com/photo-1611273426858-450d8e3c9fce?w=800&q=80" 
                 style="height: 250px; width: 100%; object-fit: cover; border-radius: 16px;">
            <div class="image-overlay-text" style="position: absolute; bottom: 20px; left: 20px;">
                <h4 style="color: white; margin: 0;">🌱 Every Action Counts</h4>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Make sustainable choices today
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### <span class='icon-animated'>🎯</span> System Status", unsafe_allow_html=True)
        st.success("✅ Smart Response Engine: Active")
        st.info("🧠 Enhanced Query Parser: Enabled")
        if llm:
            st.success("🤗 HuggingFace LLM: Connected")
        else:
            st.warning("⚠️ HuggingFace LLM: Offline Mode")
        
        st.markdown("#### <span class='icon-animated'>📊</span> Quick Stats", unsafe_allow_html=True)
        st.info(f"🌍 Global Target:\n < 6 kg CO₂/person/day by 2030")
        st.success(f"🌳 Tree Impact:\n1 tree = 21 kg CO₂/year")
    
    # Display Chat History
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("## <span class='icon-animated'>💬</span> Conversation History", unsafe_allow_html=True)
        for msg in st.session_state.conversation_history:
            if msg['role'] == 'user':
                st.markdown(f"""<div class="chat-message user-message"><strong>👤 You:</strong><br>{msg['content']}</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="chat-message assistant-message"><strong>🤖 AI Assistant:</strong><br>{msg['content']}</div>""", unsafe_allow_html=True)
    
    # Process Query
    # Check if triggered by example button or submit button
    should_process = (submit_button and user_query) or (st.session_state.trigger_submit and st.session_state.user_query)
    
    if should_process:
        # Reset trigger
        st.session_state.trigger_submit = False
        
        # Use the query from session state if triggered by example
        query_to_process = st.session_state.user_query if st.session_state.user_query else user_query
        
        st.session_state.conversation_history.append({'role': 'user', 'content': query_to_process})
        
        with st.spinner("🤖 AI is analyzing your query..."):
            # Get relevant tips from vector store
            relevant_tips = retrieve_relevant_tips(query_to_process, embedding_model, collection)
            
            # Generate intelligent response
            ai_response, current_activity, alternatives = generate_smart_response(
                query_to_process, CO2_DATA, relevant_tips
            )
            
            st.session_state.conversation_history.append({'role': 'assistant', 'content': ai_response})
            st.session_state.queries_count += 1
            
            # Calculate savings if applicable
            if current_activity and alternatives:
                best_alt = min(alternatives, key=lambda x: x['Avg_CO2_Emission(kg/day)'])
                savings = current_activity['Avg_CO2_Emission(kg/day)'] - best_alt['Avg_CO2_Emission(kg/day)']
                if savings > 0:
                    st.session_state.total_savings += savings
        
        st.markdown("---")
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("## <span class='icon-animated'>📋</span> AI Response", unsafe_allow_html=True)
        st.markdown(ai_response, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed visualizations if activity found
        if current_activity and alternatives:
            st.markdown("---")
            st.markdown("## <span class='icon-animated'>📊</span> Detailed Impact Analysis", unsafe_allow_html=True)
            
            best_alt = min(alternatives, key=lambda x: x['Avg_CO2_Emission(kg/day)'])
            savings = current_activity['Avg_CO2_Emission(kg/day)'] - best_alt['Avg_CO2_Emission(kg/day)']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Emissions", f"{current_activity['Avg_CO2_Emission(kg/day)']} kg/day")
            with col2:
                st.metric("Best Alternative", f"{best_alt['Avg_CO2_Emission(kg/day)']} kg/day", delta=f"-{savings:.2f} kg")
            with col3:
                reduction_pct = (savings / current_activity['Avg_CO2_Emission(kg/day)']) * 100 if current_activity['Avg_CO2_Emission(kg/day)'] > 0 else 0
                st.metric("Reduction", f"{reduction_pct:.1f}%", delta=f"-{savings:.2f} kg/day")
            
            st.plotly_chart(create_comparison_chart(current_activity, alternatives[:3]), use_container_width=True)
            
            if savings > 0:
                annual_savings = savings * 365
                trees_saved = annual_savings / 21
                st.markdown(f"""
                <div class="impact-banner">
                    <h2><span class='icon-animated'>🌟</span> Your Annual Impact Potential</h2>
                    <p style="font-size: 1.3rem; margin: 1rem 0; color: white;">
                        By switching to <strong>{best_alt['Activity']}</strong>, you could save:
                    </p>
                    <div style="display: flex; justify-content: center; gap: 3rem; margin-top: 1.5rem; flex-wrap: wrap;">
                        <div><div style="font-size: 3rem; font-weight: 800;">{annual_savings:.0f}</div><div style="font-size: 1.1rem; opacity: 0.95;">kg CO₂ per year</div></div>
                        <div><div style="font-size: 3rem; font-weight: 800;">{int(trees_saved)}</div><div style="font-size: 1.1rem; opacity: 0.95;">Trees Equivalent</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    elif submit_button:
        st.warning("⚠️ Please enter a question to get started!")
    
    st.markdown("---")
    
    # Dashboard Metrics
    st.markdown("## <span class='icon-animated'>📊</span> Environmental Dashboard", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card-pro"><span class="metric-icon icon-animated">🌍</span><div class="metric-value-pro">{st.session_state.queries_count}</div><div class="metric-label-pro">Queries Made</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card-pro"><span class="metric-icon icon-animated">💚</span><div class="metric-value-pro">{st.session_state.total_savings:.1f}</div><div class="metric-label-pro">kg CO₂ Saved</div></div>""", unsafe_allow_html=True)
    with col3:
        trees = st.session_state.total_savings * 365 / 21
        st.markdown(f"""<div class="metric-card-pro"><span class="metric-icon icon-animated">🌳</span><div class="metric-value-pro">{int(trees)}</div><div class="metric-label-pro">Trees Equivalent</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card-pro"><span class="metric-icon icon-animated">📈</span><div class="metric-value-pro">{len(CO2_DATA)}</div><div class="metric-label-pro">Activities Tracked</div></div>""", unsafe_allow_html=True)
    
    # Footer Section
    st.markdown('<div class="additional-resources-section">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #047857; font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 2rem;"><span class="icon-animated">📚</span> Additional Resources</h2>', unsafe_allow_html=True)
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.markdown("### 🎯 Category Breakdown")
        st.plotly_chart(create_pie_chart(CO2_DATA), use_container_width=True, key="footer_pie")
    with footer_col2:
        st.markdown("### 🔧 System Features")
        st.markdown("""
        <div class="tech-item"><span class="tech-icon icon-animated">🧠</span><span>Smart Query Parser</span></div>
        <div class="tech-item"><span class="tech-icon icon-animated">📊</span><span>Real-time Analysis</span></div>
        <div class="tech-item"><span class="tech-icon icon-animated">🗄️</span><span>Vector Search Engine</span></div>
        <div class="tech-item"><span class="tech-icon icon-animated">🤗</span><span>AI-Powered Insights</span></div>
        <div class="tech-item"><span class="tech-icon icon-animated">📈</span><span>Impact Visualization</span></div>
        """, unsafe_allow_html=True)
    with footer_col3:
        st.markdown("### <span class='icon-animated'>🎯</span> Quick Facts", unsafe_allow_html=True)
        st.info("🌍 **Global Target**\n\nLess than 6 kg CO₂ per person per day by 2030")
        st.success("🌳 **Tree Impact**\n\n1 tree absorbs ~21 kg CO₂ per year")
        if st.button("🔄 Reset Dashboard", use_container_width=True, key="footer_reset"):
            st.session_state.total_savings = 0
            st.session_state.queries_count = 0
            st.session_state.history = []
            st.session_state.conversation_history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Copyright Footer
    st.markdown("""
    <div class="copyright-footer">
        <p class="copyright-text">
            © 2025 CO₂ Reduction Platform | Powered by AI 🤖
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
