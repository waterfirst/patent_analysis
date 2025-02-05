import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re

# Download required NLTK data
nltk.download("stopwords")


API_KEY = st.secrets["USPTO_API_KEY"]


def search_patents(keywords, per_page=1000):
    query = {"_and": [{"_text_phrase": {"patent_abstract": kw}} for kw in keywords]}

    fields = [
        "patent_number",
        "patent_date",
        "patent_title",
        "patent_abstract",
        "patent_firstnamed_assignee_country",
        "patent_type",
    ]

    url = "https://api.patentsview.org/patents/query"
    headers = {"X-Api-Key": API_KEY}

    params = {
        "q": json.dumps(query),
        "f": json.dumps(fields),
        "o": json.dumps({"per_page": per_page}),
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return {"patents": []}


def process_patent_data(data):
    if not data.get("patents"):
        return pd.DataFrame()

    patents = []
    for patent in data["patents"]:
        patent_entry = {
            "Patent Number": patent.get("patent_number", ""),
            "Date": patent.get("patent_date", ""),
            "Title": patent.get("patent_title", ""),
            "Country": patent.get("patent_firstnamed_assignee_country", ""),
            "Type": patent.get("patent_type", ""),
            "Abstract": patent.get("patent_abstract", ""),
        }
        patents.append(patent_entry)

    return pd.DataFrame(patents)


def create_visualizations(df):
    # Add Year column
    df["Year"] = pd.to_datetime(df["Date"]).dt.year

    # Yearly patents bar chart
    yearly_patents = df["Year"].value_counts().sort_index()
    fig_yearly = px.bar(
        x=yearly_patents.index,
        y=yearly_patents.values,
        title="Patents by Year",
        labels={"x": "Year", "y": "Number of Patents"},
    )
    st.plotly_chart(fig_yearly)

    # Country patents bar chart
    country_patents = df["Country"].value_counts().head(10)
    fig_country = px.bar(
        x=country_patents.index,
        y=country_patents.values,
        title="Top 10 Countries by Number of Patents",
        labels={"x": "Country", "y": "Number of Patents"},
    )
    st.plotly_chart(fig_country)

    # Word Cloud
    stop_words = set(stopwords.words("english"))
    additional_stop_words = {"method", "device", "system", "apparatus", "invention"}
    stop_words.update(additional_stop_words)

    # Combine all abstracts
    text = " ".join(df["Abstract"].fillna(""))

    # Clean text
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = " ".join(filtered_words)

    # Create word cloud
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", max_words=100
    ).generate(text)

    # Display word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)


def main():
    st.title("Patent Search and Analysis Tool")

    # Input for keywords
    keyword_input = st.text_input("Enter keywords (comma-separated)")
    search_button = st.button("Search Patents")

    if search_button and keyword_input:
        keywords = [k.strip() for k in keyword_input.split(",")]

        with st.spinner("Searching patents..."):
            results = search_patents(keywords)
            df = process_patent_data(results)

            if df.empty:
                st.warning("No patents found matching the criteria")
            else:
                st.success(f"Found {len(df)} patents")
                st.dataframe(df)

                # Visualizations
                create_visualizations(df)

                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"patents_{timestamp}.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
