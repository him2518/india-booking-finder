import streamlit as st
import json
import os
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Official India Booking Linker", page_icon="🇮🇳")

# --- 1. LOAD DATABASE (Safe Path Fix) ---
def load_links():
    # This ensures we find the file no matter where Streamlit runs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'verified_links.json')
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ Critical Error: Could not find 'verified_links.json'. Please upload it to GitHub.")
        st.stop()

# --- 2. INITIALIZE AI ---
if "gemini_api_key" not in st.secrets:
    st.error("⚠️ Gemini Key Missing! Add it to Streamlit Secrets.")
    st.stop()

genai.configure(api_key=st.secrets["gemini_api_key"])

# --- 3. SMART MODEL SELECTOR (Fixes 404 Error) ---
def get_best_model():
    """
    Dynamically finds a working model to prevent 'NotFound' errors.
    """
    try:
        # Ask Google what models are available for this API Key
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Priority list (Newest & Fastest first)
        preferences = [
            'models/gemini-1.5-flash',
            'models/gemini-1.5-flash-latest',
            'models/gemini-1.5-pro',
            'models/gemini-pro',
            'models/gemini-1.0-pro'
        ]
        
        # Pick the first preferred model that actually exists
        for pref in preferences:
            if pref in available_models:
                return pref
        
        # Fallback: Just take the first available model if none match
        return available_models[0] if available_models else 'models/gemini-pro'
        
    except Exception as e:
        # If listing fails, force a standard model
        return 'models/gemini-pro'

# --- 4. THE AI BRAIN (INTENT MATCHER) ---
def find_matching_service(user_query, links_db):
    """
    Uses Gemini to match the user's vague query to a specific key in our JSON.
    """
    keys_list = list(links_db.keys())
    
    # Get a working model dynamically
    active_model = get_best_model()
    
    prompt = f"""
    You are a precise classification agent.
    
    Database Keys: {keys_list}
    User Query: "{user_query}"
    
    Task:
    1. Analyze the User Query.
    2. Identify which "Database Key" best matches the user's intent.
    3. If the user is asking for something NOT in the list, return "None".
    4. Return ONLY the exact key string (or "None"). Do not add any explanation.
    """
    
    try:
        model = genai.GenerativeModel(active_model)
        response = model.generate_content(prompt)
        cleaned_key = response.text.strip().replace('"', '').replace("'", "")
        return cleaned_key
    except Exception as e:
        st.error(f"AI Error ({active_model}): {e}")
        return "None"

# --- 5. UI LAYOUT ---
st.title("🇮🇳 Sarkari Link Finder")
st.markdown("### The Safe Way to Book Travel in India")
st.write("Don't get scammed by fake websites. We provide **only verified government links**.")

# Search Bar
query = st.text_input("What do you want to book?", placeholder="e.g. Train to Delhi, Taj Mahal ticket, Ferry to Havelock")

if st.button("🔍 Find Official Link", type="primary"):
    if not query:
        st.warning("Please type a request first!")
    else:
        with st.spinner("Searching verified government database..."):
            # Load Data
            links_db = load_links()
            
            # AI Logic
            matched_key = find_matching_service(query, links_db)
            
            st.divider()
            
            # Result Display
            if matched_key in links_db:
                result = links_db[matched_key]
                
                st.success(f"✅ Verified Service Found: **{result['service_name']}**")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Authority:** {result['authority']}")
                    st.info(f"💡 **Tip:** {result['tips']}")
                
                with col2:
                    st.link_button("🔗 GO TO OFFICIAL SITE", result['url'])
                    
                st.caption(f"Destination URL: {result['url']}")
                
            else:
                st.error("❌ Service Not in Our Verified Database")
                st.write(f"We currently only track: {', '.join(list(links_db.keys()))}")
                st.write("Try asking for 'Trains', 'Taj Mahal', or 'Kerala Ferry'.")

# --- FOOTER ---
st.divider()
st.markdown("🔒 *This tool does not handle payments. It simply redirects you to the official government portal.*")
