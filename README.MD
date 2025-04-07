# Recommender System

A recommendation system that suggests SHL assessments based on job descriptions using sentence embeddings, LLMs (Gemini), and cosine similarity. Built with a FastAPI backend and Streamlit frontend.

📦 Features
Web Scraping: Collects SHL test info (name, duration, type, job level, etc.)
Preprocessing: Cleans data and generates SBERT embeddings (all-MiniLM-L6-v2)
LLM Parsing: Uses Gemini to extract filters from natural language
Semantic Recommendation: Ranks relevant tests using cosine similarity
Evaluation: Computes Recall@K and MAP@K on sample queries
Deployment: FastAPI API + Streamlit UI

🧰 Tech Stack
Backend: FastAPI, Uvicorn
Frontend: Streamlit
LLM: Gemini via Google Generative AI SDK
NLP: SentenceTransformers
Others: LangChain, dotenv, Selenium, BeautifulSoup

🚀 Quickstart
# Install dependencies
pip install -r requirements.txt
# Set your Gemini API key
export GEMINI_API_KEY=your_api_key
# Run preprocessing
python data_preprocessing.py
# Launch frontend
streamlit run src/frontend.py

🗂 Project Structure

├── data/                         # SHL data & evaluation queries
├── src/
│   ├── backend.py                # FastAPI backend
│   ├── frontend.py               # Streamlit UI
│   ├── recommender.py            # Cosine similarity recommendation logic
│   ├── llm_processor.py          # Gemini-powered query parsing
│   ├── data_preprocessing.py     # Cleans + embeds SHL data
│   └── evaluation.py             # Evaluation metrics
├── web/
│   └── web_scraping.ipynb        # Web scraping SHL assessments
├── requirements.txt
└── README.md


