# scraper.py
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io

# Function to scrape product descriptions and images from a website
def scrape_product_data(url):
    product_descriptions = []
    product_images = []

    # Send HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all product descriptions
        descriptions = soup.find_all("h2", class_="productitem--title")
        for desc in descriptions:
            product_descriptions.append(desc.text.strip())

        # Find all product images
        product_items = soup.find_all("div", class_="productitem")
        for item in product_items:
            img_tag = item.find("img", class_="productitem--image-primary")
            if img_tag:
                img_url = img_tag['src']
                # Check if the URL starts with "http" or "//"
                if img_url.startswith("//"):
                    img_url = "https:" + img_url  # Prepend "https:" if the URL starts with "//"
                product_images.append(img_url)

    return product_descriptions, product_images

# streamlit_app.py
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords
import nltk
nltk.download('stopwords')

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
        clustering_option = st.radio("Clustering Based On", ("Text", "Image"))

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
                    st.write(cluster_descriptions[i])
                    st.image(cluster_images[i], caption='Product Image', width=200)  # Adjust width as needed
                with col2:
                    st.write(cluster_descriptions[i + images_per_row])
                    st.image(cluster_images[i + images_per_row], caption='Product Image', width=200)  # Adjust width as needed
                with col3:
                    st.write(cluster_descriptions[i + 2 * images_per_row])
                    st.image(cluster_images[i + 2 * images_per_row], caption='Product Image', width=200)  # Adjust width as needed

if __name__ == "__main__":
    main()