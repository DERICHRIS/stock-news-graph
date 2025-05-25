import streamlit as st
import requests
import datetime
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import uuid
import os
import spacy
from transformers import pipeline

# ====== YOUR NEWSAPI KEY HERE ======
NEWS_API_KEY = "e412a2cc2af94813bb55ae8a0be094ce"  # Replace with your NewsAPI key

# Load FinBERT financial sentiment model
sentiment_model = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Load spaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# --- Streamlit UI ---
st.set_page_config(page_title="Recursive Stock News Graph", layout="wide")
st.title("Recursive Company Relationship Graph")
st.subheader("Enter a company name. We’ll extract connected companies recursively from news up to a specified depth.")

company_name = st.text_input("Start Company", placeholder="e.g. Apple")
max_depth = st.slider("Depth of graph (hops)", min_value=1, max_value=6, value=3)

# --- Step 1: Fetch News ---
def fetch_news(company):
    query = f'"{company}" AND (investment OR partnership OR acquisition OR collaboration OR stock OR shares)'
    from_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
    to_date = datetime.datetime.now().strftime('%Y-%m-%d')
    url = (
        f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}"
        f"&language=en&sortBy=publishedAt&pageSize=50&apiKey={NEWS_API_KEY}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        return []

# --- Step 2: Extract Related Companies with NER and DEBUG ---
def extract_related_companies(articles, current_company):
    related = set()
    key_phrases = ["invested in", "acquired", "partnered with", "merged with", "signed a deal with"]

    for article in articles:
        content = article.get("content") or article.get("description") or ""
        lowered = content.lower()

        # DEBUG: show title for each article
        st.text(f"📰 Article: {article.get('title', 'No title')}")

        if True:  # REMOVE this line and enable filter below for production
        # if any(phrase in lowered for phrase in key_phrases):
            doc = nlp(content)
            for ent in doc.ents:
                if ent.label_ == "ORG" and ent.text.lower() != current_company.lower():
                    related.add(ent.text.strip())

    st.write(f"🔗 Related companies from {current_company}: {related}")
    return list(related)[:5]  # Limit to top 5

# --- Step 3: Sentiment Analysis ---
def analyze_sentiment(text):
    if not text:
        return "neutral", 0.0
    result = sentiment_model(text)[0]
    label_map = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral"
    }
    return label_map.get(result['label'].lower(), "neutral"), result['score']

# --- Step 4: Build Recursive Graph ---
def build_recursive_graph(start_company, max_depth):
    G = nx.Graph()
    visited = set()
    queue = [(start_company, 0)]

    while queue:
        company, depth = queue.pop(0)
        if company in visited or depth > max_depth:
            continue
        visited.add(company)

        articles = fetch_news(company)
        st.write(f"📥 Fetched {len(articles)} articles for {company}")

        related_companies = extract_related_companies(articles, company)

        for related in related_companies:
            if related not in visited:
                queue.append((related, depth + 1))
            G.add_edge(company, related)

    return G

# --- Step 5: Display Graph ---
def show_graph(G):
    if len(G.nodes) <= 1:
        st.warning("⚠️ Graph is empty or only the root node was added. Try increasing depth or using a broader company.")
        st.text(f"Graph nodes: {list(G.nodes)}")
        return

    net = Network(height="700px", width="100%", bgcolor="#111111", font_color="white")
    net.from_nx(G)
    unique_id = str(uuid.uuid4())
    html_path = f"graph_{unique_id}.html"
    net.write_html(html_path)
    with open(html_path, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=720)
    os.remove(html_path)

# --- Button ---
if st.button("Build Recursive Graph"):
    if not company_name.strip():
        st.warning("Please enter a valid company name.")
    else:
        st.info(f"Building graph up to depth {max_depth} starting from {company_name}...")
        G = build_recursive_graph(company_name.strip(), max_depth)
        show_graph(G)
        st.success("Graph generated.")
