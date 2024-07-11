from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cluster_and_rank_by_category(df, max_d, best_method):
    """
    This function will cluster and rank products by category based on their descriptions.

    Args:
    - df (DataFrame): DataFrame containing products list.
    - max_d (float): Maximum distance threshold for clustering.

    Returns:
    - df (DataFrame): DataFrame with added 'Cluster' and 'Ranking' columns.
    """
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_description']).toarray()

    linked = linkage(X, method=best_method)

    # determine clusters based on the specified max distance
    df['Cluster'] = fcluster(linked, max_d, criterion='distance')

    # sort by clusters and assign rankings
    df = df.sort_values('Cluster')
    df['Ranking'] = range(1, len(df) + 1)
    
    return df


def measure_best_distance(df):
    """
    Measure the best max distance and linkage method for clustering using silhouette score.

    Args:
    - df (DataFrame): DataFrame containing products.

    Returns:
    - overall_best_max_d (float): Best max distance found for clustering.
    - best_method (str): Best linkage method found.
    """
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_description']).toarray()
    
    methods = ['ward', 'complete', 'average', 'single']
    overall_best_sil_score = -1
    overall_best_clusters = None
    overall_best_max_d = None
    best_method = None

    # Iterate through different linkage methods
    for method in methods:
        linked = linkage(X, method)

        # Search for the best max_distance between vectors
        max_d_values = np.arange(0.5, 5.1, 0.1)
        best_sil_score = -1
        best_clusters = None
        best_max_d = None

        # Iterate through possible max distances to find the best one
        for max_d in max_d_values:
            clusters = fcluster(linked, max_d, criterion='distance')
            
            if len(set(clusters)) > 1:  
                sil_score = silhouette_score(X, clusters)
                
                # Evaluate using Silhouette Score
                if sil_score > best_sil_score:
                    best_sil_score = sil_score
                    best_clusters = clusters
                    best_max_d = max_d

        # Update overall best results if a higher one is found
        if best_clusters is not None:
            df['Cluster'] = best_clusters
            if best_sil_score > overall_best_sil_score:
                overall_best_sil_score = best_sil_score
                overall_best_clusters = best_clusters
                overall_best_max_d = best_max_d
                best_method = method
        else:
            logger.warning(f'Linkage method: {method}, no valid clusters found.')

    # Return the best max_d and the best method found
    logger.info(f'Best max distance found: {overall_best_max_d:2g}')
    logger.info(f'Best linkage method found: {best_method}')
    return overall_best_max_d, best_method


