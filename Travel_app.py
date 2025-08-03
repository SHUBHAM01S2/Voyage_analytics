import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import warnings

# Setup
warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('/home/shubham-sharma/Downloads/Voyage_Analytics/MLOPS Project/Data/hotels.csv')  # Change to your path if needed
    df.drop(['travelCode', 'userCode'], axis=1, inplace=True)
    df.rename(columns={'name': 'Hotel_name', 'place': 'Places'}, inplace=True)
    df['features'] = (
        df['Hotel_name'].astype(str) + ' ' +
        df['Places'].astype(str) + ' ' +
        df['days'].astype(str) + ' days ' +
        df['price'].astype(str) + ' price ' +
        df['total'].astype(str) + ' total'
    )
    return df

# Vectorize features
@st.cache_resource
def compute_similarity(df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    feature_matrix = cv.fit_transform(df['features']).toarray()
    cos_sim = cosine_similarity(feature_matrix)
    return cos_sim

# Recommendation logic
def recommend_hotels(input_value, df, cos_sim, num_recommendations=5):
    input_value = input_value.strip()

    # Hotel name case
    if input_value in df['Hotel_name'].values:
        st.info(f"🔍 You entered a Hotel Name: '{input_value}'")
        hotel_index = df[df['Hotel_name'] == input_value].index[0]
        input_hotel_name = df.loc[hotel_index, 'Hotel_name']
        similar_hotels = list(enumerate(cos_sim[hotel_index]))
        similar_hotels = sorted(similar_hotels, key=lambda x: x[1], reverse=True)
        filtered = [i for i in similar_hotels if df.loc[i[0], 'Hotel_name'] != input_hotel_name]
        seen = set()
        top_indices = []
        for idx, _ in filtered:
            name = df.loc[idx, 'Hotel_name']
            if name not in seen:
                seen.add(name)
                top_indices.append(idx)
            if len(top_indices) >= num_recommendations:
                break
        return df.iloc[top_indices][['Hotel_name', 'Places']]

    # Place name case
    elif input_value in df['Places'].values:
        st.info(f"🌍 You entered a Place: '{input_value}'")
        place_indices = df[df['Places'] == input_value].index.tolist()
        avg_similarity = np.mean(cos_sim[place_indices], axis=0)
        similar_hotels = list(enumerate(avg_similarity))
        similar_hotels = sorted(similar_hotels, key=lambda x: x[1], reverse=True)
        filtered = [
            i for i in similar_hotels
            if df.loc[i[0], 'Places'] != input_value
        ]
        seen_hotels = set()
        seen_places = set()
        top_indices = []
        for idx, _ in filtered:
            name = df.loc[idx, 'Hotel_name']
            place = df.loc[idx, 'Places']
            if (name, place) not in seen_hotels and place not in seen_places:
                seen_hotels.add((name, place))
                seen_places.add(place)
                top_indices.append(idx)
            if len(top_indices) >= num_recommendations:
                break
        return df.iloc[top_indices][['Hotel_name', 'Places']]

    else:
        return f"❌ No hotel or place found for: '{input_value}'"

# Streamlit UI
def main():
    st.set_page_config(page_title="Hotel Recommender", layout="centered")
    st.title("🏨 Hotel Recommendation System")

    df = load_data()
    cos_sim = compute_similarity(df)

    input_value = st.text_input("Enter a Hotel Name or a Place Name")
    num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("Get Recommendations"):
        if input_value:
            results = recommend_hotels(input_value, df, cos_sim, num_recommendations)
            if isinstance(results, str):
                st.warning(results)
            else:
                st.success("Here are your recommendations:")
                st.dataframe(results)
        else:
            st.warning("Please enter a hotel or place name to continue.")

if __name__ == '__main__':
    main()
