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

st.set_page_config(page_title="Recursive Stock News Graph", layout="wide")
st.title("Recursive Company Sentiment Graph")
st.subheader("Enter a company name. The graph shows connected companies with sentiment-colored edges from news.")

company_name = st.text_input("Start Company", placeholder="e.g. Apple")
max_depth = st.slider("Depth of graph (hops)", min_value=1, max_value=6, value=3)

# Step 1: Fetch News
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
    return []

# Step 2: Extract Related Companies

def extract_related_companies(articles, current_company):
    related = []
    key_phrases = ["invested in", "acquired", "partnered with", "merged with", "signed a deal with"]
    for idx, article in enumerate(articles):
        content = article.get("content") or article.get("description") or ""
        lowered = content.lower()
        if True:  # Temporarily disabled context filter
            doc = nlp(content)
            for ent in doc.ents:
                if ent.label_ == "ORG" and ent.text.lower() != current_company.lower():
                    sentiment, _ = analyze_sentiment(content)
                    related.append((ent.text.strip(), sentiment, idx, article.get("title", "No title")))
    return related

# Step 3: Sentiment Analysis

def analyze_sentiment(text):
    if not text:
        return "neutral", 0.0
    result = sentiment_model(text)[0]
    return result['label'].lower(), result['score']

# Step 4: Build Recursive Graph with Sentiment Edges

def build_recursive_graph(start_company, max_depth):
    G = nx.Graph()
    visited = set()
    queue = [(start_company, 0)]
    all_articles = {}

    while queue:
        company, depth = queue.pop(0)
        if company in visited or depth > max_depth:
            continue
        visited.add(company)

        articles = fetch_news(company)
        st.write(f"📥 {len(articles)} articles for {company}")
        related_data = extract_related_companies(articles, company)

        for related_company, sentiment, idx, title in related_data:
            if related_company not in visited:
                queue.append((related_company, depth + 1))

            color = {"positive": "green", "negative": "red", "neutral": "white"}.get(sentiment, "white")
            G.add_node(company, label=company)
            G.add_node(related_company, label=related_company)
            G.add_edge(company, related_company, title=f"[{idx}] {title}", color=color)
            all_articles[idx] = title

    return G, all_articles

# Step 5: Show Graph

def show_graph(G):
    if len(G.nodes) <= 1:
        st.warning("⚠️ Graph is empty or only the root node exists.")
        return

    net = Network(height="700px", width="100%", bgcolor="#111111", font_color="white")
    for node in G.nodes(data=True):
        net.add_node(node[0], label=node[1].get("label", node[0]))

    for source, target, data in G.edges(data=True):
        net.add_edge(source, target, title=data.get("title", ""), color=data.get("color", "gray"))

    html_id = f"graph_{uuid.uuid4().hex}.html"
    net.write_html(html_id)
    with open(html_id, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=720)
    os.remove(html_id)

# Streamlit logic

if st.button("Build Recursive Graph"):
    if not company_name.strip():
        st.warning("Please enter a valid company name.")
    else:
        st.info(f"Building graph up to depth {max_depth} from '{company_name}'...")
        G, article_map = build_recursive_graph(company_name.strip(), max_depth)
        show_graph(G)

        if article_map:
            st.markdown("---")
            st.subheader("News Articles (Indexed)")
            for idx, title in sorted(article_map.items()):
                st.markdown(f"**[{idx}]** {title}")
