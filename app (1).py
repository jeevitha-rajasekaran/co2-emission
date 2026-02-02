"""
CO₂ Reduction AI Agent - Enhanced with Charts After Every Query + PDF Export
COMPLETE CORRECTED VERSION
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
import base64
from io import BytesIO

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    h1, h2, h3 {
        font-weight: 700 !important;
        color: #1f2937 !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    .stTextInput input {
        border-radius: 12px !important;
        border: 2px solid #e5e7eb !important;
        padding: 12px 16px !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput input:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #10b981 !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #6b7280 !important;
    }

    .css-1d391kg {
        background: white !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }

    .stMarkdown {
        color: #374151 !important;
    }

    code {
        background: #f3f4f6 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        color: #10b981 !important;
    }

    hr {
        margin: 24px 0 !important;
        border: none !important;
        border-top: 1px solid #e5e7eb !important;
    }

    .element-container {
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA (CORRECTED WITH AC ALTERNATIVES) ====================
CO2_DATA = [
    # Transport
    {"Activity": "Car (Petrol, 20 km)", "Avg_CO2_Emission(kg/day)": 4.6, "Category": "Transport", "Icon": "🚗"},
    {"Activity": "Bus (20 km)", "Avg_CO2_Emission(kg/day)": 1.2, "Category": "Transport", "Icon": "🚌"},
    {"Activity": "Bicycle (20 km)", "Avg_CO2_Emission(kg/day)": 0.0, "Category": "Transport", "Icon": "🚴"},

    # Cooling (SEPARATED FROM HOUSEHOLD - FIX FOR AC ALTERNATIVES)
    {"Activity": "AC usage (8 hrs/day)", "Avg_CO2_Emission(kg/day)": 6.0, "Category": "Cooling", "Icon": "❄️"},
    {"Activity": "AC efficient use (24°C, 6 hrs/day)", "Avg_CO2_Emission(kg/day)": 3.5, "Category": "Cooling", "Icon": "🌡️"},
    {"Activity": "Ceiling Fan (8 hrs/day)", "Avg_CO2_Emission(kg/day)": 0.4, "Category": "Cooling", "Icon": "🌀"},
    {"Activity": "Natural Ventilation", "Avg_CO2_Emission(kg/day)": 0.0, "Category": "Cooling", "Icon": "🪟"},

    # Lighting (SEPARATED FROM HOUSEHOLD)
    {"Activity": "LED Bulb (5 hrs/day)", "Avg_CO2_Emission(kg/day)": 0.05, "Category": "Lighting", "Icon": "💡"},
    {"Activity": "Old Bulb (5 hrs/day)", "Avg_CO2_Emission(kg/day)": 0.2, "Category": "Lighting", "Icon": "🔆"},

    # Food
    {"Activity": "Meat-based diet", "Avg_CO2_Emission(kg/day)": 7.0, "Category": "Food", "Icon": "🍖"},
    {"Activity": "Vegetarian diet", "Avg_CO2_Emission(kg/day)": 2.0, "Category": "Food", "Icon": "🥗"},

    # Lifestyle
    {"Activity": "Online shopping (1)", "Avg_CO2_Emission(kg/day)": 1.0, "Category": "Lifestyle", "Icon": "📦"},
    {"Activity": "Local shopping (1)", "Avg_CO2_Emission(kg/day)": 0.3, "Category": "Lifestyle", "Icon": "🛍️"},
]

SUSTAINABILITY_TIPS = [
    "For transport, switching from petrol cars to public buses can reduce CO2 emissions by up to 74%. Buses emit approximately 1.2 kg CO2 per 20 km compared to 4.6 kg for petrol cars.",
    "Cycling is the most eco-friendly transport option with zero CO2 emissions. It's ideal for distances under 10 km and also provides health benefits.",
    "Carpooling reduces individual carbon footprint significantly. Sharing a ride with 3 colleagues reduces per-person emissions from 4.6 kg to about 1.5 kg CO2 per day.",
    "Electric vehicles (EVs) and hybrid models can reduce transport emissions by 60-80% compared to traditional petrol vehicles.",
    "LED bulbs use 75% less energy than traditional incandescent bulbs. Switching from old bulbs to LED can reduce emissions from 0.2 kg to 0.05 kg CO2 per day.",
    "Air conditioning is a major household energy consumer. Using AC efficiently (setting temperature to 24°C, reducing usage to 6 hrs/day, regular maintenance) can reduce emissions from 6.0 kg to 3.5 kg CO2 per day. Ceiling fans are excellent low-emission alternatives at just 0.4 kg CO2 per day.",
    "Plant-based diets have significantly lower carbon footprints. Vegetarian diets emit about 2.0 kg CO2 per day compared to 7.0 kg for meat-based diets - a 71% reduction.",
    "Local shopping reduces packaging and transportation emissions. It emits 0.3 kg CO2 compared to 1.0 kg for online shopping due to reduced delivery logistics."
]

# ==================== IMPROVED QUERY PARSING (UPDATED FOR NEW CATEGORIES) ====================
def parse_query_category(query: str) -> str:
    query_lower = query.lower()

    food_keywords = ['food', 'diet', 'meat', 'vegetarian', 'vegan', 'eat', 'eating', 'meal', 'plant-based', 'consume', 'consumption']
    lifestyle_keywords = ['shop', 'shopping', 'online', 'local', 'purchase', 'buy', 'buying']

    # UPDATED: Separate cooling and lighting keywords
    cooling_keywords = [' ac ', 'air condition', 'air-condition', 'a/c', 'ac usage', 'ac use', 'cooling', 'fan', 'ceiling fan', 'ventilation']
    lighting_keywords = ['bulb', 'light', 'led', 'lamp', 'lighting']

    transport_keywords = [' car ', ' drive', 'driving', 'petrol', ' bus ', 'bicycle', ' bike ', 'transport', 'commute', 'travel', 'vehicle', 'car,', 'car.', 'car?']

    if any(kw in query_lower for kw in food_keywords):
        return "Food"
    elif any(kw in query_lower for kw in lifestyle_keywords):
        return "Lifestyle"
    elif any(kw in query_lower for kw in cooling_keywords):
        return "Cooling"  # NEW
    elif any(kw in query_lower for kw in lighting_keywords):
        return "Lighting"  # NEW
    elif any(kw in query_lower for kw in transport_keywords):
        return "Transport"

    return "General"

def find_activity_from_query(query: str, CO2_DATA: list) -> dict:
    query_lower = query.lower()
    category = parse_query_category(query)

    # Food category
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

    # Lifestyle category
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

    # Cooling category (NEW - FIX FOR AC)
    if category == "Cooling":
        if 'ac' in query_lower or 'air condition' in query_lower or 'a/c' in query_lower:
            for item in CO2_DATA:
                if 'AC usage' in item['Activity']:
                    return item
        if 'fan' in query_lower:
            for item in CO2_DATA:
                if 'Fan' in item['Activity']:
                    return item
        if 'ventilation' in query_lower or 'natural' in query_lower:
            for item in CO2_DATA:
                if 'Ventilation' in item['Activity']:
                    return item
        cooling_items = [item for item in CO2_DATA if item['Category'] == 'Cooling']
        if cooling_items:
            return max(cooling_items, key=lambda x: x['Avg_CO2_Emission(kg/day)'])

    # Lighting category (NEW)
    if category == "Lighting":
        if 'led' in query_lower:
            for item in CO2_DATA:
                if 'LED' in item['Activity']:
                    return item
        if 'old' in query_lower or 'traditional' in query_lower:
            for item in CO2_DATA:
                if 'Old Bulb' in item['Activity']:
                    return item
        lighting_items = [item for item in CO2_DATA if item['Category'] == 'Lighting']
        if lighting_items:
            return max(lighting_items, key=lambda x: x['Avg_CO2_Emission(kg/day)'])

    # Transport category
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

# ==================== INTELLIGENT RESPONSE GENERATOR (FIXED SYNTAX + LONG-TERM GOALS) ====================
def generate_smart_response(query: str, CO2_DATA: list, relevant_tips: list) -> tuple:
    category = parse_query_category(query)
    current_activity = find_activity_from_query(query, CO2_DATA)

    # Category icons
    category_icons = {
        "Transport": "🚗",
        "Cooling": "❄️",
        "Lighting": "💡",
        "Food": "🥗",
        "Lifestyle": "🛍️",
        "General": "🌍"
    }

    if not current_activity:
        cat_name = category.lower() if category != "General" else "sustainability"
        icon = category_icons.get(category, category_icons['General'])
        response = f"**{icon} Sustainability Guidance**\n\n"
        response += f"Based on your query about {cat_name}, here are key recommendations:\n\n"

        if relevant_tips:
            for idx, tip in enumerate(relevant_tips[:3], 1):
                response += f"{idx}. {tip}\n\n"
        else:
            response += "• Focus on reducing emissions in transport, household energy, and food choices\n"
            response += "• Small daily changes can lead to significant annual CO₂ reductions\n"
            response += "• Consider alternatives that align with your lifestyle and location\n"

        return response, None, []

    # Get alternatives from the SAME category
    category_items = [item for item in CO2_DATA if item['Category'] == current_activity['Category']]
    alternatives = [item for item in category_items if item != current_activity]
    alternatives.sort(key=lambda x: x['Avg_CO2_Emission(kg/day)'])

    # Build response - FIXED STRING FORMATTING TO AVOID SYNTAX ERROR
    icon = current_activity['Icon']
    activity_name = current_activity['Activity']
    category_name = current_activity['Category']
    emissions = current_activity['Avg_CO2_Emission(kg/day)']

    response = f"**{icon} Your Current Activity Analysis**\n\n"
    response += f"**Current Activity:** {activity_name} {icon}\n"
    response += f"**Category:** {category_name}\n"
    response += f"**Daily CO₂ Emissions:** {emissions} kg\n\n"
    response += "---\n\n"
    response += "**💡 Recommended Alternatives:**\n\n"

    for idx, alt in enumerate(alternatives[:3], 1):
        emission_diff = current_activity['Avg_CO2_Emission(kg/day)'] - alt['Avg_CO2_Emission(kg/day)']
        if emission_diff > 0:
            reduction_pct = (emission_diff / current_activity['Avg_CO2_Emission(kg/day)']) * 100
            annual_savings = emission_diff * 365
            trees = int(annual_savings / 21)

            alt_name = alt['Activity']
            alt_icon = alt['Icon']
            alt_emission = alt['Avg_CO2_Emission(kg/day)']

            response += f"{idx}. **{alt_name}** {alt_icon}\n"
            response += f"   • Reduces to: {alt_emission} kg CO₂/day\n"
            response += f"   • Daily savings: {emission_diff:.2f} kg CO₂\n"
            response += f"   • Reduction: {reduction_pct:.0f}%\n"
            response += f"   • Annual impact: {annual_savings:.0f} kg CO₂ (≈ {trees} trees planted)\n\n"

    # ALWAYS show Long-term Sustainability Goal (NEW FEATURE)
    response += "\n---\n\n"
    response += "**📚 Long-term Sustainability Goal:**\n\n"

    # Category-specific long-term goals
    category_goals = {
        "Cooling": "**Long-term target:** Reduce cooling energy consumption by 50% through efficient AC use (24°C setting, proper insulation, smart scheduling) combined with natural ventilation. This sustainable approach can save over 1,000 kg CO₂ annually and significantly reduce energy bills.",

        "Lighting": "**Long-term target:** Complete transition to LED lighting across all rooms and outdoor spaces. A full household switch reduces lighting emissions by 75% and saves 50+ kg CO₂ per year while providing better quality light.",

        "Transport": "**Long-term target:** Reduce personal vehicle dependency by 30% through a mix of carpooling, public transport, and cycling for distances under 5 km. This multi-modal approach can save over 500 kg CO₂ annually while improving health and reducing costs.",

        "Food": "**Long-term target:** Adopt a flexitarian diet by reducing meat consumption by 50% and increasing plant-based meals. This balanced approach can save over 900 kg CO₂ per year while maintaining nutritional needs and reducing food costs.",

        "Lifestyle": "**Long-term target:** Shift 70% of purchases to local shopping, reduce packaging waste, and adopt a circular economy mindset (reuse, repair, recycle). This can save 250+ kg CO₂ annually while supporting local businesses and reducing waste."
    }

    # Use relevant tips if available, otherwise use category-specific goals
    if relevant_tips and len(relevant_tips) > 0:
        response += relevant_tips[0]
    else:
        goal = category_goals.get(current_activity['Category'], 
                                  "**Long-term target:** Reduce your overall carbon footprint by 20% in the next 6 months through consistent sustainable choices across transport, energy, and lifestyle.")
        response += goal

    return response, current_activity, alternatives

# ==================== HUGGINGFACE LLM CONFIGURATION ====================
def initialize_llm():
    if not LANGCHAIN_AVAILABLE:
        return None

    try:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
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

# ==================== ENHANCED CHART FUNCTIONS ====================
def create_comparison_chart(current_activity, alternatives):
    """Create beautiful horizontal bar chart"""
    activities = [current_activity['Activity']] + [alt['Activity'] for alt in alternatives]
    emissions = [current_activity['Avg_CO2_Emission(kg/day)']] + [alt['Avg_CO2_Emission(kg/day)'] for alt in alternatives]
    colors = ['#ef4444'] + ['#10b981'] * len(alternatives)

    fig = go.Figure()

    for i, (activity, emission, color) in enumerate(zip(activities, emissions, colors)):
        fig.add_trace(go.Bar(
            y=[activity],
            x=[emission],
            orientation='h',
            marker=dict(color=color, line=dict(color='white', width=2)),
            text=f"{emission} kg",
            textposition='outside',
            textfont=dict(size=14, color='#1f2937', family='Inter', weight='bold'),
            hovertemplate=f"<b>{activity}</b><br>CO2: {emission} kg/day<extra></extra>",
            name=activity,
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            text="<b>CO2 Emissions Comparison</b>",
            font=dict(size=24, color='#1f2937', family='Inter'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="CO2 Emissions (kg/day)",
            showgrid=True,
            gridwidth=1,
            gridcolor='#f3f4f6',
            showline=True,
            linewidth=2,
            linecolor='#e5e7eb'
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            autorange='reversed'
        ),
        height=400,
        plot_bgcolor='rgba(240, 253, 244, 0.3)',
        paper_bgcolor='white',
        font=dict(family='Inter', size=12),
        margin=dict(t=80, b=60, l=200, r=100),
        barmode='overlay'
    )

    return fig

def create_pie_chart(data):
    df = pd.DataFrame(data)
    category_emissions = df.groupby('Category')['Avg_CO2_Emission(kg/day)'].sum().reset_index()

    fig = px.pie(
        category_emissions,
        values='Avg_CO2_Emission(kg/day)',
        names='Category',
        color_discrete_sequence=['#10b981', '#3b82f6', '#ef4444', '#f59e0b', '#8b5cf6', '#ec4899'],
        hole=0.5
    )

    fig.update_traces(textposition='outside', textinfo='label+percent', marker=dict(line=dict(color='white', width=3)))
    fig.update_layout(height=400, showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1), font=dict(family='Inter', size=12), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig

# ==================== PDF GENERATION FUNCTIONS ====================
def generate_pdf_report(user_query, ai_response, current_activity, alternatives, savings):
    """Generate HTML-based PDF report"""
    if not current_activity or not alternatives:
        return None

    best_alt = min(alternatives, key=lambda x: x['Avg_CO2_Emission(kg/day)'])
    annual_savings = savings * 365
    trees_saved = int(annual_savings / 21)
    reduction_pct = (savings / current_activity['Avg_CO2_Emission(kg/day)']) * 100 if savings > 0 else 0

    # Clean AI response for PDF
    clean_response = re.sub('<[^<]+?>', '', ai_response)
    clean_response = clean_response.replace('**', '')

    # Create chart HTML
    activities_data = [
        (current_activity['Activity'], current_activity['Avg_CO2_Emission(kg/day)'], '#ef4444')
    ] + [
        (alt['Activity'], alt['Avg_CO2_Emission(kg/day)'], '#10b981')
        for alt in alternatives[:3]
    ]

    max_emission = max([item[1] for item in activities_data])
    chart_html = ''
    for activity, emission, color in activities_data:
        bar_width = (emission / max_emission) * 100
        chart_html += f"""
        <div style="margin-bottom: 10px;">
            <div style="font-weight: 600; margin-bottom: 4px; font-size: 12px;">{activity}</div>
            <div style="background: #f3f4f6; border-radius: 8px; height: 30px; position: relative;">
                <div style="background: {color}; height: 100%; width: {bar_width}%; border-radius: 8px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px;">
                    <span style="color: white; font-weight: 600; font-size: 12px;">{emission} kg</span>
                </div>
            </div>
        </div>
        """

    # Generate alternatives table
    alternatives_html = ''
    for idx, alt in enumerate(alternatives[:3], 1):
        emission_diff = current_activity['Avg_CO2_Emission(kg/day)'] - alt['Avg_CO2_Emission(kg/day)']
        alt_reduction_pct = (emission_diff / current_activity['Avg_CO2_Emission(kg/day)']) * 100 if emission_diff > 0 else 0
        alt_annual_savings = emission_diff * 365

        alternatives_html += f"""
        <div style="background: #f9fafb; padding: 16px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #10b981;">
            <h4 style="margin: 0 0 8px 0; color: #1f2937;">{idx}. {alt['Activity']} {alt['Icon']}</h4>
            <ul style="margin: 0; padding-left: 20px; color: #4b5563; font-size: 13px;">
                <li>Emissions: {alt['Avg_CO2_Emission(kg/day)']} kg CO2/day</li>
                <li>Daily savings: {emission_diff:.2f} kg CO2</li>
                <li>Reduction: {alt_reduction_pct:.0f}%</li>
                <li>Annual impact: {alt_annual_savings:.0f} kg CO2</li>
            </ul>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            body {{ font-family: 'Inter', sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; color: #1f2937; line-height: 1.6; }}
            .header {{ text-align: center; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 32px; border-radius: 16px; margin-bottom: 32px; }}
            .header h1 {{ margin: 0 0 8px 0; font-size: 32px; }}
            .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
            .section {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
            .section h2 {{ color: #10b981; font-size: 20px; margin: 0 0 16px 0; border-bottom: 2px solid #10b981; padding-bottom: 8px; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 24px 0; }}
            .metric {{ text-align: center; background: #f9fafb; padding: 16px; border-radius: 12px; border: 2px solid #e5e7eb; }}
            .metric-value {{ font-size: 28px; font-weight: 700; color: #10b981; margin-bottom: 4px; }}
            .metric-label {{ font-size: 12px; color: #6b7280; font-weight: 500; }}
            .chart-section {{ margin: 24px 0; }}
            .footer {{ text-align: center; padding: 24px; background: #f9fafb; border-radius: 12px; margin-top: 32px; color: #6b7280; font-size: 12px; }}
            .query-box {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 16px; border-radius: 8px; margin-bottom: 16px; font-style: italic; color: #1e40af; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌍 CO2 Reduction Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>

        <div class="section">
            <h2>📊 Your Query</h2>
            <div class="query-box">{user_query}</div>
        </div>

        <div class="section">
            <h2>📈 Current Activity</h2>
            <p><strong>Activity:</strong> {current_activity['Activity']} {current_activity['Icon']}</p>
            <p><strong>Category:</strong> {current_activity['Category']}</p>
            <p><strong>Daily Emissions:</strong> {current_activity['Avg_CO2_Emission(kg/day)']} kg CO2</p>
        </div>

        <div class="section">
            <h2>📉 Emissions Comparison</h2>
            <div class="chart-section">
                {chart_html}
            </div>
        </div>

        <div class="section">
            <h2>💡 Recommended Alternatives</h2>
            {alternatives_html}
        </div>

        <div class="section">
            <h2>🎯 Impact Summary</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value">{savings:.1f} kg</div>
                    <div class="metric-label">Daily CO2 Saved</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{annual_savings:.0f} kg</div>
                    <div class="metric-label">Annual CO2 Saved</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{trees_saved}</div>
                    <div class="metric-label">Trees Equivalent</div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p><strong>🤖 AI-Powered Environmental Intelligence</strong></p>
            <p>This report was generated using smart AI analysis and sustainability data.</p>
        </div>
    </body>
    </html>
    """

    return html_content

def get_pdf_download_link(html_content):
    """Generate download link for PDF report"""
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="co2_reduction_report.html" style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; margin-top: 16px;">📥 Download Report (HTML)</a>'
    return href

# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; background: white; padding: 2rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
        <h1 style='color: #10b981; margin-bottom: 0.5rem;'>🌍 CO2 Reduction Platform</h1>
        <p style='color: #6b7280; font-size: 18px; margin: 0;'>AI-Powered Environmental Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize embedding system
    embedding_model, collection = initialize_vector_store()

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💾 Total CO2 Saved", f"{st.session_state.total_savings:.1f} kg", "Lifetime")
    with col2:
        st.metric("🔍 Queries Analyzed", st.session_state.queries_count, "Total")
    with col3:
        trees_equivalent = int(st.session_state.total_savings / 21)
        st.metric("🌳 Trees Equivalent", trees_equivalent, "Planted")

    st.markdown("---")

    # User input
    st.markdown("""
    <div style='background: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
        <h3 style='color: #1f2937; margin-top: 0;'>💬 Tell us about your daily activity</h3>
        <p style='color: #6b7280; margin-bottom: 1rem;'>Ask about transport, food, energy usage, or any daily activity...</p>
    </div>
    """, unsafe_allow_html=True)

    user_query = st.text_input(
        "Enter your query:",
        placeholder="e.g., 'I use AC for 8 hours every day' or 'I drive a car 20km daily'",
        label_visibility="collapsed",
        key="main_input"
    )

    col_submit, col_clear = st.columns([1, 4])
    with col_submit:
        submit_button = st.button("🔍 Analyze", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.history = []
            st.rerun()

    # Process query
    if submit_button and user_query:
        with st.spinner("🤖 Analyzing your carbon footprint..."):
            # Retrieve relevant tips
            relevant_tips = retrieve_relevant_tips(user_query, embedding_model, collection) if embedding_model and collection else []

            # Generate response
            response, current_activity, alternatives = generate_smart_response(user_query, CO2_DATA, relevant_tips)

            # Calculate savings
            if current_activity and alternatives:
                best_alternative = min(alternatives, key=lambda x: x['Avg_CO2_Emission(kg/day)'])
                savings = current_activity['Avg_CO2_Emission(kg/day)'] - best_alternative['Avg_CO2_Emission(kg/day)']
                if savings > 0:
                    st.session_state.total_savings += savings
            else:
                savings = 0

            st.session_state.queries_count += 1

            # Add to history
            st.session_state.conversation_history.append({
                "query": user_query,
                "response": response,
                "current_activity": current_activity,
                "alternatives": alternatives,
                "savings": savings
            })

    # Display results
    if st.session_state.conversation_history:
        latest = st.session_state.conversation_history[-1]

        st.markdown("---")
        st.markdown("## 🤖 AI Assistant Response")

        # Display response in a nice card
        st.markdown(f"""
        <div style='background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 1.5rem;'>
            {latest['response']}
        </div>
        """, unsafe_allow_html=True)

        # Show chart if we have activity data
        if latest['current_activity'] and latest['alternatives']:
            st.markdown("## 📊 Visual Comparison")
            chart = create_comparison_chart(latest['current_activity'], latest['alternatives'])
            st.plotly_chart(chart, use_container_width=True)

            # PDF download
            st.markdown("## 📄 Export Report")
            pdf_html = generate_pdf_report(
                latest['query'],
                latest['response'],
                latest['current_activity'],
                latest['alternatives'],
                latest['savings']
            )
            if pdf_html:
                st.markdown(get_pdf_download_link(pdf_html), unsafe_allow_html=True)

    # Sidebar with category breakdown
    with st.sidebar:
        st.markdown("## 📈 Category Breakdown")
        pie_chart = create_pie_chart(CO2_DATA)
        st.plotly_chart(pie_chart, use_container_width=True)

        st.markdown("## 📚 Quick Facts")
        st.info("🌳 21 kg CO2 = 1 tree planted")
        st.info("🚗 Average car: 4.6 kg CO2/day")
        st.info("🥗 Vegetarian diet: 71% less CO2")

if __name__ == "__main__":
    main()
