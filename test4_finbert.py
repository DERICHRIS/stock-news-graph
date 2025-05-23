import streamlit as st
import requests
import datetime
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import uuid
import os
from transformers import pipeline

# ====== YOUR NEWSAPI KEY HERE ======
NEWS_API_KEY = "e412a2cc2af94813bb55ae8a0be094ce"  # Replace with your NewsAPI key
# ===================================

# Load FinBERT financial sentiment model
sentiment_model = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Sentiment Graph", layout="centered")
st.title("Stock Sentiment Graph")
st.subheader("Enter a company name to analyze recent news sentiment and visualize it.")

company_name = st.text_input("Company Name", placeholder="e.g. Apple, Microsoft, Tesla")

# --- Step 1: Fetch News ---
def fetch_news(company):
    query = f'"{company}" AND (stock OR shares OR company OR business OR technology)'
    from_date = (datetime.datetime.now() - datetime.timedelta(days=15)).strftime('%Y-%m-%d')
    to_date = datetime.datetime.now().strftime('%Y-%m-%d')

    url = (
        f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}"
        f"&language=en&sortBy=publishedAt&pageSize=50&apiKey={NEWS_API_KEY}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        st.error("Failed to fetch news. Check your API key or quota.")
        return []

# --- Step 2: Sentiment Analysis using FinBERT ---
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

# --- Step 3: Relevance Filtering ---
def is_article_relevant(article, company):
    fields = [article.get('title', ''), article.get('description', ''), article.get('content', '')]
    full_text = " ".join([f if f is not None else "" for f in fields]).lower()
    keywords = [company.lower(), f"{company.lower()} inc", f"{company.lower()} stock"]
    return any(kw in full_text for kw in keywords)

# --- Step 4: Build Sentiment Graph ---
def build_sentiment_graph(articles, center_label="Company"):
    G = nx.Graph()

    # Central node
    G.add_node("center", label=center_label, color="blue", size=30)

    for idx, article in enumerate(articles):
        title = article['title']
        url = article.get('url', '#')
        sentiment, _ = analyze_sentiment(article['description'] or article['content'] or "")
        color = {
            "positive": "green",
            "negative": "red",
            "neutral": "white"
        }[sentiment]
        edge_color = color
        node_id = f"news_{idx}"
        G.add_node(node_id, label=str(idx), title=title, color="white", href=url)
        G.add_edge("center", node_id, color=edge_color)

    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)

    net.set_options('''{
      "nodes": {
        "shape": "dot",
        "font": {
          "multi": true
        }
      },
      "interaction": {
        "tooltipDelay": 200,
        "hideEdgesOnDrag": true
      }
    }''')

    unique_id = str(uuid.uuid4())
    output_file = f"sentiment_graph_{unique_id}.html"
    net.write_html(output_file)

    with open(output_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=620)

    os.remove(output_file)

# --- Button Logic ---
if st.button("Analyze News"):
    if not company_name.strip():
        st.warning("Please enter a valid company name.")
    else:
        st.success(f"Fetching news and analyzing sentiment for: {company_name}")
        news_articles = fetch_news(company_name)

        if news_articles:
            st.write("Filtered News with Sentiment")
            relevant_articles = []

            for article in news_articles:
                if not is_article_relevant(article, company_name):
                    continue

                relevant_articles.append(article)

            if relevant_articles:
                st.write("Sentiment Graph")
                build_sentiment_graph(relevant_articles, center_label=company_name)

                for idx, article in enumerate(relevant_articles):
                    content = article['description'] or article['content'] or ""
                    sentiment, score = analyze_sentiment(content)

                    emoji = {
                        "positive": "[POSITIVE]",
                        "negative": "[NEGATIVE]",
                        "neutral": "[NEUTRAL]"
                    }[sentiment]

                    st.markdown(f"{emoji} **[{idx}] [{article['title']}]({article.get('url', '#')})**")
                    st.caption(article['publishedAt'])
                    st.write(f"Sentiment: {sentiment.capitalize()} ({score:.2f})")
                    st.write(content if content else "No description available.")
                    st.markdown("---")
            else:
                st.info("No highly relevant articles found — try a broader name like 'Apple Inc'.")
        else:
            st.info("No articles found.")
