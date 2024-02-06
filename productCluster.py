import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords and punkt tokenizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Function to perform K-means clustering on text data
def text_clustering(descriptions, num_clusters=3):
    # Tokenize the descriptions and remove stop words
    stop_words = set(stopwords.words('english'))
    tokenized_descriptions = []
    for desc in descriptions:
        words = word_tokenize(desc)
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
        tokenized_descriptions.append(' '.join(filtered_words))

    # Vectorize the tokenized descriptions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_descriptions)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    labels = kmeans.labels_
    return labels

def main():
    st.title("Product Clustering App")

    # Sidebar for user input
    st.sidebar.header("Scraping Parameters")
    url = st.sidebar.text_input("Enter URL of the website to scrape")
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)

    if st.sidebar.button("Scrape Data"):
        product_descriptions, product_images = scrape_product_data(url)

        # Perform clustering
        # clustering_option = st.radio("Clustering Based On", ("Text", "Image"))
        
        clustering_option = "Text"

        if clustering_option == "Text":
            # Perform text clustering
            labels = text_clustering(product_descriptions, num_clusters)
        elif clustering_option == "Image":
            # Perform image clustering
            labels = image_clustering(product_images, num_clusters)

        # Display clustering results
        st.write(f"Number of clusters: {num_clusters}")
        
        # Display descriptions and images for each cluster
        for cluster_num in range(num_clusters):
            st.subheader(f"Cluster {cluster_num + 1}")  # Add a subtitle for the cluster
            
            # Get descriptions and images for the current cluster
            cluster_descriptions = [desc for desc, label in zip(product_descriptions, labels) if label == cluster_num]
            cluster_images = [img for img, label in zip(product_images, labels) if label == cluster_num]
            
            num_columns = 3  # Number of columns for image display
            images_per_row = len(cluster_images) // num_columns

            for i in range(images_per_row):
                # Create a new row for each group of images
                col1, col2, col3 = st.columns(num_columns)
                with col1:
                    display_product_card(cluster_descriptions[i], cluster_images[i])
                with col2:
                    display_product_card(cluster_descriptions[i + images_per_row], cluster_images[i + images_per_row])
                with col3:
                    display_product_card(cluster_descriptions[i + 2 * images_per_row], cluster_images[i + 2 * images_per_row])

def display_product_card(description, image_url):
    # Use HTML and CSS to create a card-like component
    st.markdown(
        f"""
        <div style="border: 1px solid #ccc; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
            <h3>{description}</h3>
            <img src="{image_url}" style="width: 200px;">
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
