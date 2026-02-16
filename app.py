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
KNOWN_GOV_RELATED_DOMAINS = {
    "irctc.co.in",
    "ksrtc.in",
    "soutickets.in",
    "maavaishnodevi.org",
}
SERVICE_TERMS = {
    "bus", "train", "ferry", "ticket", "tickets", "booking", "book", "reservation",
    "darshan", "entry", "visa", "passport", "yatra", "registration", "tourist",
}
ALIASES = {
    "uttarpradesh": "uttar pradesh",
    "uttar-pradesh": "uttar pradesh",
    "andamanandnicobar": "andaman nicobar",
    "karnatakastate": "karnataka",
}


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


def normalize_host(url):
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


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
    discovered_host = normalize_host(discovered_url)
    discovered_title = top_result.get("title", "Official Service Portal").strip()

    # Update existing record if same host already exists.
    for key, row in links_db.items():
        existing_host = normalize_host(row.get("url", ""))
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
        "authority": "Government Portal (AI-assisted web discovered)",
        "coverage": location_query.strip() or "India",
        "keywords": build_keywords(service_query, location_query, discovered_title),
        "url": discovered_url,
        "tips": "Auto-added from AI/heuristic government-likelihood check. Verify latest instructions on the portal.",
    }
    return "created", new_key


def normalize(text):
    normalized_text = text.lower()
    for raw, replacement in ALIASES.items():
        normalized_text = normalized_text.replace(raw, replacement)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", normalized_text)).strip()


def tokenize(text):
    return [t for t in normalize(text).split() if t and t not in STOPWORDS]


def local_match_service(user_query, links_db):
    query_tokens = set(tokenize(user_query))
    query_location_tokens = {t for t in query_tokens if t not in SERVICE_TERMS}
    best_key = None
    best_score = 0.0

    for key, data in links_db.items():
        if not isinstance(data, dict):
            continue
        key_phrase = key.replace("_", " ")
        service_texts = [
            key_phrase,
            data.get("service_name", ""),
            data.get("authority", ""),
            " ".join(data.get("keywords", [])),
        ]
        coverage_text = data.get("coverage", "")
        service_tokens = set(tokenize(" ".join(service_texts)))
        coverage_tokens = set(tokenize(coverage_text))
        candidate_tokens = service_tokens | coverage_tokens

        overlap_score = (
            len(query_tokens & service_tokens) / max(len(service_tokens), 1)
            if query_tokens
            else 0.0
        )
        coverage_overlap_score = (
            len(query_location_tokens & coverage_tokens) / max(len(coverage_tokens), 1)
            if query_location_tokens
            else 0.0
        )
        similarity_score = max(
            SequenceMatcher(None, normalize(user_query), normalize(candidate)).ratio()
            for candidate in service_texts + [coverage_text]
            if candidate
        )
        # If user gave location hints but this service has no location overlap, penalize it.
        location_mismatch_penalty = 0.0
        if query_location_tokens and coverage_tokens and not (query_location_tokens & coverage_tokens):
            location_mismatch_penalty = 0.20

        score = (
            (0.50 * overlap_score)
            + (0.25 * similarity_score)
            + (0.35 * coverage_overlap_score)
            - location_mismatch_penalty
        )

        if score > best_score:
            best_key = key
            best_score = score

    return best_key, best_score


def get_api_key():
    try:
        secret_value = st.secrets.get("gemini_api_key")
        if secret_value:
            return secret_value
    except Exception:
        # No secrets.toml configured; continue with environment fallback.
        pass
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
    query_location_tokens = {t for t in tokenize(user_query) if t not in SERVICE_TERMS}

    if local_key and local_score >= 0.45:
        return local_key, "local"

    if ai_enabled:
        ai_key = ai_match_service(user_query, links_db)
        if ai_key:
            return ai_key, "ai"

    if local_key and local_score >= 0.25 and not query_location_tokens:
        return local_key, "local-low-confidence"

    return None, "none"


def is_priority_official_url(url):
    host = normalize_host(url)
    return host.endswith(OFFICIAL_SUFFIXES) or host in KNOWN_GOV_RELATED_DOMAINS


def is_allowed_web_result(url):
    host = normalize_host(url)
    blocked_hosts = (
        "youtube.com",
        "instagram.com",
        "facebook.com",
        "x.com",
        "twitter.com",
        "linkedin.com",
        "wikipedia.org",
        "tripadvisor.",
        "booking.com",
    )
    if any(blocked in host for blocked in blocked_hosts):
        return False
    return bool(host)


def heuristic_government_score(url, title, snippet, query):
    host = normalize_host(url)
    text_blob = normalize(f"{title} {snippet} {query} {host}")
    score = 0.0

    if host.endswith(OFFICIAL_SUFFIXES):
        score += 0.8
    if host in KNOWN_GOV_RELATED_DOMAINS:
        score += 0.65

    positive_terms = [
        "government", "govt", "ministry", "department", "directorate", "official",
        "state transport", "tourism board", "shrine board", "municipal corporation",
        "ministry of", "government of india", "govt of",
    ]
    negative_terms = [
        "agent", "private", "sponsored", "ad", "broker", "travel agency", "coupon",
        "offer", "discount",
    ]

    for term in positive_terms:
        if term in text_blob:
            score += 0.06
    for term in negative_terms:
        if term in text_blob:
            score -= 0.08

    return min(max(score, 0.0), 1.0)


def ai_government_assessment(url, title, snippet, query, ai_enabled):
    if not ai_enabled:
        return None
    active_model = get_best_model()
    if not active_model:
        return None
    prompt = f"""
Classify if this website is an official government portal in India.
Return ONLY JSON in one line with keys:
is_government (true/false), confidence (0 to 1), reason (short text).

URL: {url}
Title: {title}
Snippet: {snippet}
User Query: {query}
"""
    try:
        model = genai.GenerativeModel(active_model)
        response = model.generate_content(prompt)
        raw = (getattr(response, "text", "") or "").strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        candidate = match.group(0) if match else raw
        data = json.loads(candidate)
        conf = float(data.get("confidence", 0))
        conf = min(max(conf, 0.0), 1.0)
        is_gov = bool(data.get("is_government", False))
        reason = str(data.get("reason", "")).strip()
        return {"is_government": is_gov, "confidence": conf, "reason": reason}
    except Exception:
        return None


def assess_government_likelihood(url, title, snippet, query, ai_enabled):
    heuristic_score = heuristic_government_score(url, title, snippet, query)
    ai_result = None
    if heuristic_score >= 0.2:
        ai_result = ai_government_assessment(url, title, snippet, query, ai_enabled)

    if ai_result:
        ai_score = ai_result["confidence"] if ai_result["is_government"] else (1 - ai_result["confidence"]) * 0.3
        final_score = (0.55 * heuristic_score) + (0.45 * ai_score)
        accepted = final_score >= 0.62 and (heuristic_score >= 0.45 or ai_result["is_government"])
    else:
        final_score = heuristic_score
        accepted = final_score >= 0.7

    return {
        "accepted": accepted,
        "final_score": round(final_score, 3),
        "heuristic_score": round(heuristic_score, 3),
        "ai_result": ai_result,
    }


@st.cache_data(show_spinner=False, ttl=3600)
def web_fallback_search(user_query, ai_enabled):
    if DDGS is None:
        return []

    search_query = f"{user_query} official government portal india booking registration"
    seen = set()
    accepted = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=14)
            for row in results:
                url = row.get("href") or row.get("url") or ""
                title = (row.get("title") or "").strip()
                snippet = (row.get("body") or "").strip()
                if not url or url in seen or not is_allowed_web_result(url):
                    continue
                seen.add(url)
                assessment = assess_government_likelihood(url, title, snippet, user_query, ai_enabled)
                if not assessment["accepted"]:
                    continue
                accepted.append(
                    {
                        "title": title or "Official portal",
                        "url": url,
                        "snippet": snippet,
                        "official": is_priority_official_url(url),
                        "gov_score": assessment["final_score"],
                        "ai_reason": (assessment["ai_result"] or {}).get("reason", ""),
                    }
                )
    except Exception:
        return []

    accepted.sort(key=lambda x: (-x["gov_score"], 0 if x["official"] else 1, x["title"].lower()))
    return accepted[:5]


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
    st.caption("Running in self-contained mode (no API key required). Web fallback uses stricter heuristics without AI.")

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
                if not isinstance(result, dict):
                    st.error("Matched entry is not a valid service record.")
                    st.stop()
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
                    with st.spinner("Searching web and checking government likelihood..."):
                        web_results = web_fallback_search(combined_query, ai_enabled)
                    if web_results:
                        st.warning(
                            "Not found in internal verified DB. Showing high-confidence government-like portals "
                            "based on domain signals and AI assessment."
                        )
                        top_result = web_results[0]
                        st.link_button(
                            f"🌐 OPEN TOP RESULT: {top_result['title']}",
                            top_result["url"],
                        )
                        st.caption(f"{top_result['url']} (government-likelihood score: {top_result['gov_score']})")
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
                            st.markdown(
                                f"- [{item['title']}]({item['url']}) "
                                f"(score: {item['gov_score']})"
                            )
                    else:
                        st.info("No high-confidence government-like results found for this query.")

st.divider()
st.markdown(
    "🔒 *This tool does not handle payments. It only redirects to official portals.*"
)
