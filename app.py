import json
import os
import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

import streamlit as st

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None


st.set_page_config(page_title="Official India Booking Linker", page_icon="🇮🇳")

STOPWORDS = {
    "a", "an", "and", "book", "booking", "for", "from", "go", "i", "in", "is",
    "me", "my", "of", "on", "online", "please", "site", "the", "ticket", "tickets", "to",
    "want", "where",
}

OFFICIAL_SUFFIXES = (".gov.in", ".nic.in")


@st.cache_data(show_spinner=False)
def load_links():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "verified_links.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Critical error: `verified_links.json` is missing from the repository.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Critical error: `verified_links.json` is not valid JSON.")
        st.stop()


def get_links_file_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "verified_links.json")


def slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:60] if slug else "unknown_service"


def build_keywords(service_query, location_query, title):
    merged = f"{service_query} {location_query} {title}".strip()
    tokens = []
    for token in tokenize(merged):
        if token not in tokens:
            tokens.append(token)
    return tokens[:12]


def save_links_db(links_db):
    json_path = get_links_file_path()
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(links_db, f, ensure_ascii=False, indent=2)
        load_links.clear()
        return True, None
    except Exception as e:
        return False, str(e)


def upsert_discovered_result(links_db, service_query, location_query, top_result):
    discovered_url = top_result["url"]
    discovered_host = urlparse(discovered_url).netloc.lower()
    discovered_title = top_result.get("title", "Official Service Portal").strip()

    # Update existing record if same host already exists.
    for key, row in links_db.items():
        existing_host = urlparse(row.get("url", "")).netloc.lower()
        if existing_host == discovered_host:
            row["service_name"] = row.get("service_name") or discovered_title
            row["authority"] = row.get("authority") or "Government Portal"
            row["coverage"] = row.get("coverage") or (location_query.strip() or "India")
            row["tips"] = row.get("tips") or "Always verify details on the official portal before payment."
            row["keywords"] = list(
                dict.fromkeys((row.get("keywords") or []) + build_keywords(service_query, location_query, discovered_title))
            )[:15]
            if row.get("url") != discovered_url:
                row["url"] = discovered_url
            return "updated", key

    base_key = slugify(f"{service_query}_{location_query}_{discovered_title}")
    new_key = base_key
    suffix = 2
    while new_key in links_db:
        new_key = f"{base_key}_{suffix}"
        suffix += 1

    links_db[new_key] = {
        "service_name": discovered_title,
        "authority": "Government Portal (web discovered)",
        "coverage": location_query.strip() or "India",
        "keywords": build_keywords(service_query, location_query, discovered_title),
        "url": discovered_url,
        "tips": "Auto-added from strict official domain search. Verify latest instructions on the portal.",
    }
    return "created", new_key


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def tokenize(text):
    return [t for t in normalize(text).split() if t and t not in STOPWORDS]


def local_match_service(user_query, links_db):
    query_tokens = set(tokenize(user_query))
    best_key = None
    best_score = 0.0

    for key, data in links_db.items():
        key_phrase = key.replace("_", " ")
        candidates = [
            key_phrase,
            data.get("service_name", ""),
            data.get("authority", ""),
            data.get("coverage", ""),
            " ".join(data.get("keywords", [])),
        ]
        candidate_tokens = set(tokenize(" ".join(candidates)))

        overlap_score = (
            len(query_tokens & candidate_tokens) / max(len(candidate_tokens), 1)
            if query_tokens
            else 0.0
        )
        similarity_score = max(
            SequenceMatcher(None, normalize(user_query), normalize(candidate)).ratio()
            for candidate in candidates
            if candidate
        )
        score = (0.65 * overlap_score) + (0.35 * similarity_score)

        if score > best_score:
            best_key = key
            best_score = score

    return best_key, best_score


def get_api_key():
    if "gemini_api_key" in st.secrets:
        return st.secrets["gemini_api_key"]
    if "GEMINI_API_KEY" in os.environ:
        return os.environ["GEMINI_API_KEY"]
    if "GOOGLE_API_KEY" in os.environ:
        return os.environ["GOOGLE_API_KEY"]
    return None


@st.cache_data(show_spinner=False)
def get_best_model():
    if genai is None:
        return None
    try:
        available_models = [
            m.name for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        preferences = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-pro",
            "models/gemini-pro",
            "models/gemini-1.0-pro",
        ]
        for pref in preferences:
            if pref in available_models:
                return pref
        return available_models[0] if available_models else "models/gemini-pro"
    except Exception:
        return "models/gemini-pro"


def ai_match_service(user_query, links_db):
    keys_list = list(links_db.keys())
    active_model = get_best_model()
    if not active_model:
        return None

    prompt = f"""
You are a classification agent.
Database Keys: {keys_list}
User Query: "{user_query}"
Return ONLY one exact Database Key from the list above, or "None".
"""
    try:
        model = genai.GenerativeModel(active_model)
        response = model.generate_content(prompt)
        output_text = getattr(response, "text", "") or ""
        candidate = output_text.strip().strip('"').strip("'")
        return candidate if candidate in links_db else None
    except Exception:
        return None


def find_matching_service(user_query, links_db, ai_enabled):
    local_key, local_score = local_match_service(user_query, links_db)

    if local_key and local_score >= 0.45:
        return local_key, "local"

    if ai_enabled:
        ai_key = ai_match_service(user_query, links_db)
        if ai_key:
            return ai_key, "ai"

    if local_key and local_score >= 0.25:
        return local_key, "local-low-confidence"

    return None, "none"


def is_priority_official_url(url):
    host = urlparse(url).netloc.lower()
    return host.endswith(OFFICIAL_SUFFIXES)


def is_allowed_web_result(url):
    host = urlparse(url).netloc.lower()
    return host.endswith(OFFICIAL_SUFFIXES)


@st.cache_data(show_spinner=False, ttl=3600)
def web_fallback_search(user_query):
    if DDGS is None:
        return []

    search_query = f"{user_query} official government portal india site:gov.in OR site:nic.in"
    seen = set()
    ranked = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=10)
            for row in results:
                url = row.get("href") or row.get("url") or ""
                title = (row.get("title") or "").strip()
                snippet = (row.get("body") or "").strip()
                if not url or url in seen or not is_allowed_web_result(url):
                    continue
                seen.add(url)
                rank_boost = 1 if is_priority_official_url(url) else 0
                ranked.append(
                    {
                        "title": title or "Official portal",
                        "url": url,
                        "snippet": snippet,
                        "official": rank_boost == 1,
                    }
                )
    except Exception:
        return []

    ranked.sort(key=lambda x: (0 if x["official"] else 1, x["title"].lower()))
    return ranked[:5]


api_key = get_api_key()
ai_enabled = bool(api_key and genai is not None)
if ai_enabled:
    try:
        genai.configure(api_key=api_key)
    except Exception:
        ai_enabled = False


st.title("🇮🇳 India Government Booking Finder")
st.markdown("### Find the official government website to book services in India")
st.write(
    "Travelers often need services like train tickets, ferry booking, bus tickets, or monument entry "
    "but don't know where to book online safely. This tool helps you find verified official portals."
)

if ai_enabled:
    st.caption("AI assist is enabled. Local matching remains active as fallback.")
else:
    st.caption("Running in self-contained mode (no API key required).")

service_query = st.text_input(
    "What service do you want to avail?",
    placeholder="e.g. train ticket, Taj Mahal entry, Kerala ferry, e-Visa, passport appointment",
)
location_query = st.text_input(
    "From where / for which place? (optional)",
    placeholder="e.g. Delhi, Agra, Kerala, Andaman, Uttarakhand, Jammu",
)
enable_web_fallback = st.checkbox(
    "If not found in database, search internet for possible official booking links",
    value=True,
)

if st.button("🔍 Find Official Link", type="primary"):
    if not service_query.strip():
        st.warning("Please type the service you want first.")
    else:
        with st.spinner("Searching verified government database..."):
            links_db = load_links()
            combined_query = f"{service_query} {location_query}".strip()
            matched_key, method = find_matching_service(combined_query, links_db, ai_enabled)
            st.divider()

            if matched_key and matched_key in links_db:
                result = links_db[matched_key]
                st.success(f"Verified service found: **{result['service_name']}**")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Authority:** {result['authority']}")
                    if result.get("coverage"):
                        st.write(f"**Applies to:** {result['coverage']}")
                    st.info(f"Tip: {result['tips']}")
                with col2:
                    st.link_button("🔗 GO TO OFFICIAL SITE", result["url"])
                st.caption(f"Destination URL: {result['url']}")
                if method == "local-low-confidence":
                    st.caption("Matched using best local guess.")
            else:
                st.error("Service not found in our verified database.")
                st.write(f"Currently tracked services: {', '.join(list(links_db.keys()))}")
                st.write(
                    "Try asking for `train ticket` (Delhi/Mumbai), `Taj Mahal` (Agra), "
                    "`Kerala ferry`, `Andaman ferry`, `e-Visa`, `passport appointment`, "
                    "`Char Dham registration`, or `Vaishno Devi yatra`."
                )
                if enable_web_fallback:
                    with st.spinner("Searching web for likely official portals..."):
                        web_results = web_fallback_search(combined_query)
                    if web_results:
                        st.warning(
                            "Not found in internal verified DB. Showing strict official-domain results "
                            "(`.gov.in` / `.nic.in`)."
                        )
                        top_result = web_results[0]
                        st.link_button(
                            f"🌐 OPEN TOP RESULT: {top_result['title']}",
                            top_result["url"],
                        )
                        st.caption(top_result["url"])
                        action, record_key = upsert_discovered_result(
                            links_db, service_query, location_query, top_result
                        )
                        saved, save_error = save_links_db(links_db)
                        if saved:
                            st.success(
                                f"Local DB auto-{action}: `{record_key}`. Future searches can use this record."
                            )
                        else:
                            st.warning(
                                "Result found but could not persist DB update in this environment. "
                                f"Error: {save_error}"
                            )
                        st.markdown("Other suggested links:")
                        for item in web_results[1:]:
                            st.markdown(f"- [{item['title']}]({item['url']})")
                    else:
                        st.info("No strict official-domain results found for this query.")

st.divider()
st.markdown(
    "🔒 *This tool does not handle payments. It only redirects to official portals.*"
)
