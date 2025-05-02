# 🧠 SHL Assessment Recommender System

A smart, semantic search-based recommender system that suggests SHL-style assessments by scraping, embedding, and evaluating job-level tests using NLP and vector similarity.

---

## 📌 Project Overview

- Scrapes assessment data (title, description, type, duration, etc.)
- Uses **Selenium** and **BeautifulSoup** for dynamic and static scraping
- Embeds data using **Sentence Transformers** or **OpenAI Embeddings**
- Implements hybrid search: local vector similarity + API-based semantic matching
- Evaluates performance using human-generated queries and IR metrics

---

## 🚀 Features

- 🕸️ Automated test scraping from SHL catalog
- 🔍 Query matching using semantic embeddings
- 📊 Recall@K and MAP@K evaluation with cosine similarity
- 💡 Streamlit UI + FastAPI backend for real-time recommendations

---

## 🛠️ Tech Stack

| Tool/Library          | Purpose                                   |
|-----------------------|-------------------------------------------|
| BeautifulSoup4        | Static HTML scraping                      |
| Selenium              | Dynamic scraping for JavaScript pages     |
| SentenceTransformers  | Embeddings for semantic similarity        |
| OpenAI                | API-based embedding & matching            |
| Streamlit             | Frontend interface                        |
| FastAPI               | Backend API                               |
| Pandas, NumPy         | Data processing                           |
| Graphviz              | Flowchart generation                      |

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/shl-recommender.git
cd shl-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
