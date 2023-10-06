import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset (replace 'combine_sales.csv' with your actual dataset file)
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.head(10000)
else:
    st.sidebar.warning("Please upload a CSV file.")
    st.stop()

# Select relevant features for recommendation
features = ['Sub Section name', 'COLOR', 'SIZE', 'STYLE', 'FABRIC']
data = df[features]
data['combined_features'] = data.apply(lambda row: ' '.join([str(row[feature]) for feature in features]), axis=1)

# Use TfidfVectorizer to convert text data to numerical vectors
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

# Compute cosine similarity between items
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

recommendations = []

# Function to get recommendations without repeating colors, styles, and fabrics
def get_recommendations(item_name, cosine_similarities, data):
    try:
        item_index = data[data['Sub Section name'] == item_name].index[0]
        similar_items = list(enumerate(cosine_similarities[item_index]))
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
        
        seen_colors = set()
        seen_styles = set()
        seen_fabrics = set()
        unique_similar_items = []
        
        for index, similarity in similar_items:
            product = data.loc[index]
            product_color = product['COLOR']
            product_style = product['STYLE']
            product_fabric = product['FABRIC']
            
            if product_color not in seen_colors and product_style not in seen_styles and product_fabric not in seen_fabrics:
                unique_similar_items.append((index, similarity))
                seen_colors.add(product_color)
                seen_styles.add(product_style)
                seen_fabrics.add(product_fabric)
                
            if len(unique_similar_items) >= 5:
                break
        
        return unique_similar_items
    except IndexError:
        return []
# Updated CSS for enhanced styling
st.markdown(
    """
    <style>
    .header-text {
        font-size: 36px;
        color: #ff5555;
        text-align: center;
        padding: 20px;
    }
    .recommendation-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius:  5px;
        margin:20px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
        width: 600;
    }
    .recommendation-text {
        text-align: center;
        font-size: 18px;
        color: #333;
        margin-bottom: 10px;
    }
    .no-recommendation {
        font-size: 24px;
        color: #ff5555;
        text-align: center;
        margin-top: 50px;
    }
    .container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Chique Clothing Company - Product Recommendations")
st.sidebar.header("Configuration")

# Option to either enter product name or choose from dropdown
option = st.sidebar.radio("Choose Option", ("Enter Product Name", "Choose Product Name"))

if option == "Enter Product Name":
    # User input for product name
    product_name = st.sidebar.text_input("Enter Product Name:")
    # Convert the input to uppercase
    product_name = product_name.upper()
    if product_name:
        recommendations = get_recommendations(product_name, cosine_sim, data)
else:
    # Dropdown to choose product name
    selected_product = st.sidebar.selectbox("Select a Product", data['Sub Section name'].unique(), index=None)
    if selected_product:
        recommendations = get_recommendations(selected_product, cosine_sim, data)

# Display recommendations in colorful cards
if recommendations:
    st.markdown("<div class='header-text'>Top 5 Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    for i, (index, similarity) in enumerate(recommendations):
        recommended_item = data.loc[index]
        st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='recommendation-text'><b>{recommended_item['Sub Section name']}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p class='recommendation-text'>Color: {recommended_item['COLOR']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='recommendation-text'>Style: {recommended_item['STYLE']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='recommendation-text'>Fabric: {recommended_item['FABRIC']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='recommendation-text'>Cosine Similarity: {similarity:.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='no-recommendation'>No recommendations available.</div>", unsafe_allow_html=True)





