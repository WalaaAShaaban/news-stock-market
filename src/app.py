import streamlit as st
from preprocessing_data import PreprocessingData
from src.models.train_model import TrainModel
from src.models.build_models.build_models import BuildModles

def main():
    st.title("📈 Stock Market Prediction News")
    st.header("Analysis of financial news on equities to improve decision-making")
    st.text("Stay updated with market trends and insights!")

    # Adding a text input for user search
    query = st.text_input("Enter a stock date between (2007-07-07 , 2024-07-08) ", "")

    # Adding a button to trigger analysis
    if st.button("The most important news for investment"):
        st.write(f"Fetching and analyzing news for: **{query}**")
        # You can integrate an API call or data processing here

    # Adding a sidebar
    st.sidebar.title("Navigation")
    st.sidebar.write("🔍 Search news")
    st.sidebar.write("📊 Stock trends")
    st.sidebar.write("📉 Market sentiment")

    preprocessing_data = PreprocessingData()
    preprocessing_data.read_data()
    preprocessing_data.save_data()

    buildModels =  BuildModles()
    buildModels.build_model()
    
if __name__ == "__main__":
    main()
