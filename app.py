import streamlit as st
import json
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Official India Booking Linker", page_icon="🇮🇳")

# Load Database
def load_links():
    with open('verified_links.json', 'r') as f:
        return json.load(f)

# Initialize AI
if "gemini_api_key" not in st.secrets:
    st.error("⚠️ Gemini Key Missing! Add it to .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=st.secrets["gemini_api_key"])

# --- THE AI BRAIN (INTENT MATCHER) ---
def find_matching_service(user_query, links_db):
    """
    Uses Gemini to match the user's vague query to a specific key in our JSON.
    """
    keys_list = list(links_db.keys())
    
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
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    cleaned_key = response.text.strip().replace('"', '').replace("'", "")
    
    return cleaned_key

# --- UI LAYOUT ---
st.title("🇮🇳 Sarkari Link Finder")
st.markdown("### The Safe Way to Book Travel in India")
st.write("Don't get scammed by fake websites. We provide **only verified government links**.")

# 1. Search Bar
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
                    # A big, obvious button
                    st.link_button("🔗 GO TO OFFICIAL SITE", result['url'])
                    
                st.caption(f"Destination URL: {result['url']}")
                
            else:
                st.error("❌ Service Not in Our Verified Database")
                st.write(f"We currently only track: {', '.join(list(links_db.keys()))}")
                st.write("Try asking for 'Trains', 'Taj Mahal', or 'Kerala Ferry'.")

# --- FOOTER ---
st.divider()
st.markdown("🔒 *This tool does not handle payments. It simply redirects you to the official government portal.*")