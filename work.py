import joblib
import pandas as pd
import numpy as np
import datetime as dt

from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2500)
pd.set_option('display.expand_frame_repr', False)



metadata = pd.read_json('games_metadata.json', lines=True)
metadata['tags'] = metadata['tags'].apply(lambda x: ', '.join(x))
metadata['tags'] = metadata['tags'].apply(lambda x: np.nan if x == '' else x)
metadata['tags'].isnull().sum()
metadata.dropna(inplace=True)
games = pd.read_csv('games.csv')
nodlcgames = games['title'].str.contains('DLC', case=False)
games = games[~nodlcgames]
noSoundtrackgames = games['title'].str.contains('Soundtrack', case=False)
games = games[~noSoundtrackgames]

games['date_release'] = pd.to_datetime(games['date_release'])
twenty_years_ago = datetime.now() - timedelta(days=365*20)
filtered_games = games[(games['positive_ratio'] >= 50) & (games['user_reviews'] >= 30) & (games['date_release'] > twenty_years_ago)]

content_recom = pd.merge(filtered_games, metadata, on='app_id')
relevant_cols = content_recom[['app_id', 'title', 'tags']]

all_tags = ','.join(relevant_cols['tags']).split(',')
tag_counts = {}
for tag in all_tags:
    if tag in tag_counts:
        tag_counts[tag] += 1
    else:
        tag_counts[tag] = 1

popular_tags = {tag: count for tag, count in tag_counts.items() if count > 2000}
sorted_data = sorted(popular_tags.items(), key=lambda x: x[1], reverse=True)
df_sorted_data_tag = pd.DataFrame(sorted_data, columns=['Tag', 'Count'])
# CSV dosyası olarak çıkarma
df_sorted_data_tag.to_csv('top_15_tag_counts.csv', index=False)


for item in sorted_data[:15]:
    print(item)



for tag in popular_tags:
    relevant_cols[tag] = relevant_cols['tags'].str.contains(tag).astype(int)
relevant_cols.columns = relevant_cols.columns.str.strip()
relevant_cols.loc[:, ~relevant_cols.columns.isin(['app_id','title', 'description', 'tags']) & ~relevant_cols.columns.isin(popular_tags)] = 0
relevant_cols.info()
relevant_cols.isnull().any()

relevant_cols = relevant_cols.loc[:, ~relevant_cols.columns.duplicated()]

drop_columns=['2D', '3D',  'Anime',  'Co-op', 'Colorful',  'Comedy',
       'Cute', 'Difficult', 'Early Access',
       'Exploration',  'Family Friendly', 'Fantasy',
       'Female Protagonist', 'First-Person', 'Free to Play', 'Funny',
       'Gore', 'Great Soundtrack', 'Horror',
        'Open World',  'Pixel Graphics',
       'Platformer',
       'Relaxing', 'Retro',
       'Sci-fi', 'Shooter',
       'Third Person',  'Violent',
       'app_id', 'tags', 'title']
features=relevant_cols.drop(columns=drop_columns, axis=1)
features.info()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', NearestNeighbors(n_neighbors=5))
])

pipeline.fit(features)


def get_recommendations(app_id, data, model_pipeline):
    game_index = data.loc[data['app_id'] == app_id].index[0]

    game_features = features.iloc[[game_index]]

    _, indices = model_pipeline.named_steps['knn'].kneighbors(
        model_pipeline.named_steps['scaler'].transform(game_features), n_neighbors=2)

    recommended_index = indices[0][1]

    return data.iloc[recommended_index]['app_id']

relevant_cols.to_csv('data.csv', index=False)




app_id = relevant_cols.sample(1)['app_id'].values[0]
recommendations = get_recommendations(app_id, relevant_cols, pipeline)

game_name = relevant_cols.loc[relevant_cols['app_id'] == app_id, ['title']].values[0]
recom_game= relevant_cols.loc[relevant_cols['app_id'] == recommendations, ['title']].values[0]

print(f"gamename: '{game_name}' recom_gamename: {recom_game}")

joblib.dump(pipeline, 'knn_game_recommender_pipeline.pkl')



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# Content-Based Filtering
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

def recommend_games(title, games_data):
    tfidf_matrix = tfidf_vectorizer.fit_transform(games_data['title'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    game_index = games_data[games_data['title'] == title].index[0]
    similar_indices = cosine_similarities[game_index].argsort()[:-6:-1]
    recommended_games = [games_data['title'].iloc[i] for i in similar_indices]
    return recommended_games

# Pipeline oluşturma
content_based_pipeline = Pipeline([
    ('tfidf_vectorizer', tfidf_vectorizer),
    ('recommendation_model', recommend_games)
])
joblib.dump(content_based_pipeline, 'content_based_pipeline.pkl')

