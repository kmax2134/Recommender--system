# SHL Assessment Recommendation System

A system that recommends SHL assessments based on job descriptions using embeddings, LLMs, and similarity matching. Built with FastAPI backend and Streamlit frontend.

---
## Part 1: Web Scraping

Scrapes SHL assessment data to build a dataset for a recommendation system.  
Collects name, URL, remote/adaptive flags, test types, duration, description, and job levels.

## Tools
`requests`, `BeautifulSoup`, `selenium`, `pandas`

## Run
```bash
python scrape_static.py      # For catalog data  
python scrape_dynamic.py     # For detailed info

## Part 2: Data Preprocessing

**File**: `data_preprocessing.py`  
- Loads raw SHL CSV data  
- Cleans and combines fields into descriptive text  
- Generates sentence embeddings using `all-MiniLM-L6-v2`  
- Saves as `processed_shl_data.pkl`

---

## Part 3: Recommendation System

**Files**:  
- `src/recommender.py`: Core logic for filtering and ranking SHL tests using cosine similarity + LLM-preprocessed filters  
- `src/llm_processor.py`: Uses Gemini API to extract structured info (duration, remote, test type, job level, etc.) from user query  
- `src/backend.py`: FastAPI endpoint `/recommend` serving SHL recommendations  
- `src/frontend.py`: Streamlit UI to input job query and display results  

---

## Part 4: Evaluation

**File**: `src/evaluation.py`  
- Contains ground truth test queries  
- Uses SBERT to compute semantic `Recall@K` and `MAP@K`  
- Evaluates recommendation performance and prints metrics  

---

## Part 5: Deployment

**FastAPI Backend**  
- Run: `python src/backend.py`  
- Endpoint: `POST /recommend` returns recommended assessments  

**Streamlit Frontend**  
- Run: `streamlit run src/frontend.py`  
- Features:  
  - Text area for job description input  
  - Optional filters (duration, job level)  
  - Evaluation section to test system on sample queries  

---

## Requirements

- Python 3.8+  
- Streamlit, FastAPI, Uvicorn  
- SentenceTransformers, LangChain, dotenv, Google Generative AI SDK  

---

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_api_key
python data_preprocessing.py
streamlit run src/frontend.py


