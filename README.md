Product Clustering and Ranking

This project clusters and ranks products based on their descriptions using hierarchical clustering and TF-IDF vectorization. It's designed to help organize and prioritize products by similarity in their descriptions.

Setup

Make sure you have Python 3.12 installed.

Installation

Create your virtual environment.

for example:
>>python -m venv venv 

Install the required packages using pip:

>>pip install -r requirements.txt

Usage

Run main.py to preprocess data, cluster products, and export results:
>>python main.py

Files

main.py: Loads data from product_list.csv, preprocesses descriptions, clusters products by category, and exports results to output.csv.

db_utils/clustering.py: Contains functions for clustering products (cluster_and_rank_by_category) and finding optimal clustering parameters (measure_best_distance).

db_utils/preprocessing.py: Provides text preprocessing functions (preprocess_text) using NLTK.

Data

Ensure product_list.csv is in the root directory. **VERY IMPORTANT** This file should contain product information with a column named description.

Notes

The project uses logging to display progress and important information during execution.
For more details on clustering and preprocessing, refer to the respective Python files in db_utils.

Example Usage

Ensure product_list.csv is populated with product data.
Run main.py to see clustered products in output.csv.
Feel free to modify and extend the code based on your specific requirements!

Reach out if you have any idea to include here: xndrsa@gmail.com