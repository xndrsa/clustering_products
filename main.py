import pandas as pd
import logging
from db_utils.preprocessing import preprocess_text
from db_utils.clustering import cluster_and_rank_by_category, measure_best_distance


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info('Loading data...')
df = pd.read_csv('product_list.csv')


logger.info('Cleaning column "description"...')
df = df.dropna(subset=['description'])
df = df.astype(str) # Convert all columns to string type

# apply preprocessing to the 'description' column
logger.info('Preprocessing data...')
df['cleaned_description'] = df['description'].apply(preprocess_text)

# check for the 'category' column, create a default category if it does not exists, get a unique list
if 'category' not in df.columns:
    df['category'] = 'default_category'
categories = df['category'].unique()

logger.info('Finding the best measure for the cluster...')
max_d , best_method = measure_best_distance(df)

# apply clustering and ranking, split by categories
df_ranked = pd.DataFrame()
for category in categories:
    df_category = df[df['category'] == category].copy()
    df_ranked_category = cluster_and_rank_by_category(df_category, max_d,best_method)
    df_ranked = pd.concat([df_ranked, df_ranked_category])

df_ranked.reset_index(drop=True, inplace=True)

# removing the temp 'cleaned_description' column and 'category' if it's a default category
logger.info('Cleaning up final DataFrame...')
df_ranked = df_ranked.drop(columns=['cleaned_description'])
if 'default_category' in categories:
    df_ranked = df_ranked.drop(columns=['category'])

# saving to csv
df_ranked.to_csv('output.csv', index=False)
logger.info('Output file exported successfully.')
