import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests
import pandas as pd
from src.evaluation import evaluate_model, ground_truth  # Ensure ground_truth is defined in evaluation.py

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("SHL Assessment Recommendation System")

# ---------------------------
# Initialize session state
# ---------------------------
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ---------------------------
# 📋 Get Recommendations
# ---------------------------
st.header("📋 Get Recommendations")

job_levels = [
    "Director", "Entry-Level", "Executive", "Front Line Manager", "General Population",
    "Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor"
]

with st.form("recommendation_form"):
    query = st.text_area("Enter job description or query:",
                         placeholder="e.g., I need a cognitive ability test for analysts under 30 minutes")
    duration_filter = st.number_input("Maximum duration in minutes (optional):", min_value=0, max_value=120, step=5)
    selected_levels = st.multiselect("Select job level(s) (optional):", job_levels)
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    payload = {
        "query": query,
        "max_duration": duration_filter if duration_filter > 0 else None,
        "job_levels": selected_levels if selected_levels else None
    }
    try:
        response = requests.post("https://recommender-system-1ckr.onrender.com/recommend", json=payload)
        if response.status_code == 200:
            st.session_state.recommendations = response.json()["recommendations"]
        else:
            st.error("Error getting recommendations. Please try again.")
            st.session_state.recommendations = []
    except Exception as e:
        st.error(f"Request error: {e}")
        st.session_state.recommendations = []

# ---------------------------
# 📄 Show Recommendations
# ---------------------------
if st.session_state.recommendations is not None:
    recommendations = st.session_state.recommendations
    if not recommendations:
        st.warning("No matching assessments found.")
    else:
        df = pd.DataFrame(recommendations)

        # Mapping full names to symbols for test types
        test_type_mapping = {
            'Ability & Aptitude': 'A',
            'Biodata & Situational Judgement': 'B',
            'Competencies': 'C',
            'Development & 360': 'D',
            'Assessment Exercises': 'E',
            'Knowledge & Skills': 'K',
            'Personality & Behavior': 'P',
            'Simulations': 'S'
        }

        def map_test_types(test_types):
            if isinstance(test_types, list):
                return ", ".join([test_type_mapping.get(t, t) for t in test_types])
            return test_type_mapping.get(test_types, test_types)

        df['Assessment'] = df.apply(lambda x: f"[{x['name']}]({x['url']})", axis=1)
        df['Remote'] = df['remote'].map({"Yes": "Yes"}).fillna("No")
        df['Adaptive'] = df['adaptive'].map({"Yes": "Yes"}).fillna("No")
        df['Duration (mins)'] = df['duration_minutes']
        df['Test Type'] = df['test_type'].apply(map_test_types)

        st.markdown("### 🗝️ Test Type Key")
        test_type_key = {
            'A': 'Ability & Aptitude',
            'B': 'Biodata & Situational Judgement',
            'C': 'Competencies',
            'D': 'Development & 360',
            'E': 'Assessment Exercises',
            'K': 'Knowledge & Skills',
            'P': 'Personality & Behavior',
            'S': 'Simulations'
        }
        key_str = ", ".join([f"**{k}** = {v}" for k, v in test_type_key.items()])
        st.markdown(key_str)

        st.markdown("### Recommended Assessments")
        st.table(df[['Assessment', 'Test Type', 'Remote', 'Adaptive', 'Duration (mins)']])

# ---------------------------
# 📊 Evaluation Section
# ---------------------------
with st.expander("📊 Evaluate System on Sample Queries"):
    k = st.slider("Select K for Recall@K / MAP@K:", 1, 10, 3, key="eval_slider")
    threshold = st.slider("Set Semantic Similarity Threshold (0.0 = lenient, 1.0 = strict):",
                          min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    if st.button("Run Evaluation", key="eval_button"):
        st.session_state.eval_results = evaluate_model(ground_truth, k=k, similarity_threshold=threshold)

    if st.session_state.eval_results:
        st.success("Evaluation completed successfully!")
        st.metric(f"Mean Recall@{k}", st.session_state.eval_results.get(f"Mean Recall@{k}", 0.0))
        st.metric(f"MAP@{k}", st.session_state.eval_results.get(f"MAP@{k}", 0.0))
