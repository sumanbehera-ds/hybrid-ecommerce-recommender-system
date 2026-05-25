import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="E-commerce Recommender",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-commerce Recommendation System")
st.caption("Hybrid recommender: GRU4Rec + Item-CF + Popularity fallback")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input User Item Sequence")

    item_sequence_text = st.text_area(
        "Enter item IDs separated by commas",
        value="325215, 259884, 216305, 342816",
        height=120
    )

    top_n = st.slider("Number of recommendations", 1, 20, 10)

    recommend_btn = st.button("Generate Recommendations", type="primary")

with col2:
    st.subheader("API Health")

    if st.button("Check API Status"):
        try:
            res = requests.get(f"{API_URL}/health", timeout=10)
            if res.status_code == 200:
                st.success("API is healthy")
                st.json(res.json())
            else:
                st.error("API returned an error")
        except Exception as e:
            st.error(f"API not reachable: {e}")

st.divider()

if recommend_btn:
    try:
        item_sequence = [
            int(item.strip())
            for item in item_sequence_text.split(",")
            if item.strip()
        ]

        payload = {
            "item_sequence": item_sequence,
            "top_n": top_n
        }

        response = requests.post(
            f"{API_URL}/recommend",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()

            st.success("Recommendations generated")

            c1, c2, c3 = st.columns(3)
            c1.metric("Input Items", len(data["input_sequence"]))
            c2.metric("Recommendations", len(data["recommendations"]))
            c3.metric("Model", data["model_used"])

            st.subheader("Recommended Item IDs")

            rec_cols = st.columns(5)

            for idx, item_id in enumerate(data["recommendations"]):
                with rec_cols[idx % 5]:
                    st.info(f"#{idx + 1}\n\nItem ID: {item_id}")

            st.subheader("Raw API Response")
            st.json(data)

        else:
            st.error(response.json())

    except ValueError:
        st.error("Invalid input. Use only comma-separated integer item IDs.")
    except Exception as e:
        st.error(f"Request failed: {e}")

st.divider()

st.subheader("Popular Items")

if st.button("Show Popular Recommendations"):
    try:
        res = requests.get(f"{API_URL}/recommend/popular?top_n=10", timeout=30)
        if res.status_code == 200:
            data = res.json()
            st.json(data)
        else:
            st.error("Failed to fetch popular items")
    except Exception as e:
        st.error(f"API error: {e}")