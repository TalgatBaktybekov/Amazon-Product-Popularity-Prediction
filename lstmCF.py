import pandas as pd
import numpy as np
import torch
import gc
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy import sparse
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def preprocess_data(df, min_reviews_first_part=5, min_reviews_second_part=1, time_split='2018-06-01'):
 
    if 'reviewTime' in df.columns:
        df['reviewTime'] = pd.to_datetime(df['reviewTime'])
    
    df['description'] = df.get('description', pd.Series([''] * len(df))).fillna('')
    df['rating'] = df.get('rating', pd.Series([np.nan] * len(df))).fillna(df['rating'].mean())
    df['vote'] = df.get('vote', pd.Series([0] * len(df))).fillna(0)
    df['brand'] = df.get('brand', pd.Series(['Unknown'] * len(df))).fillna('Unknown').astype(str)
    df['category'] = df.get('category', pd.Series(['Unknown'] * len(df))).fillna('Unknown').astype(str)
    
    # split data
    time_split = pd.to_datetime(time_split)
    df_first_part = df[df['reviewTime'] < time_split]
    df_second_part = df[df['reviewTime'] >= time_split]
    
    # Filter items
    review_counts_first = df_first_part.groupby('itemName').size()
    review_counts_second = df_second_part.groupby('itemName').size()
    valid_items = review_counts_first[review_counts_first >= min_reviews_first_part].index
    valid_items = valid_items.intersection(review_counts_second[review_counts_second >= min_reviews_second_part].index)
    df_filtered = df[df['itemName'].isin(valid_items)].copy()
    
    print(f"Filtered to {len(valid_items)} items with ≥{min_reviews_first_part} reviews before {time_split} "
          f"and ≥{min_reviews_second_part} reviews on/after {time_split}")
    return df_filtered

def create_popularity_labels(df, time_split='2018-06-01'):
    # group into weekly data
    temp_df = df.groupby(['itemName', pd.Grouper(key='reviewTime', freq='W')]).agg({
        'userName': 'count',
        'rating': 'mean',
        'vote': 'sum'
    }).rename(columns={'userName': 'review_count'}).reset_index()
    
    # define pop_score
    temp_df['pop_score'] = (0.6 * temp_df['review_count'] +
                            0.2 * temp_df['rating']/5 +
                            0.2 * np.log1p(temp_df['vote']))
    
    # splitting into first and second parts
    time_split = pd.to_datetime(time_split)
    train_df = temp_df[temp_df['reviewTime'] < time_split].copy()
    test_df = temp_df[temp_df['reviewTime'] >= time_split].copy()
    
    # Compute threshold and labels based on second part
    max_pop_score_test = test_df.groupby('itemName')['pop_score'].max().reset_index()
    # threshold = max_pop_score_test['pop_score'].quantile(0.97)
    threshold = 3.128
    labels_df = max_pop_score_test[['itemName']].copy()
    labels_df['popular'] = (max_pop_score_test['pop_score'] > threshold).astype(int)

    items = labels_df['itemName'].unique()
    train_items, test_items = train_test_split(items, test_size=0.2, random_state=42)
    
    # The dataset for the model building
    min_week = train_df['reviewTime'].min()
    all_weeks = pd.date_range(start=min_week, end=time_split - pd.Timedelta(days=1), freq='W-SUN')
    complete_index = pd.MultiIndex.from_product([items, all_weeks], names=['itemName', 'reviewTime'])
    complete_df = pd.DataFrame(index=complete_index).reset_index()
    wide_first = complete_df.merge(
        train_df[['itemName', 'reviewTime', 'review_count', 'rating', 'vote', 'pop_score']],
        on=['itemName', 'reviewTime'],
        how='left'
    ).fillna({'review_count': 0, 'rating': train_df['rating'].mean(), 'vote': 0, 'pop_score': train_df['pop_score'].mean()})
    
    review_count_wide_first = wide_first.pivot(index='itemName', columns='reviewTime', values='review_count').fillna(0)
    rating_wide_first = wide_first.pivot(index='itemName', columns='reviewTime', values='rating').fillna(train_df['rating'].mean())
    vote_wide_first = wide_first.pivot(index='itemName', columns='reviewTime', values='vote').fillna(0)
    pop_score_wide_first = wide_first.pivot(index='itemName', columns='reviewTime', values='pop_score').fillna(train_df['pop_score'].mean())
    
    return (train_df, test_df, labels_df, train_items, test_items,
            review_count_wide_first, rating_wide_first, vote_wide_first, pop_score_wide_first, threshold)

class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super(AdvancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1) 
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, 1)  
        self.sigmoid = nn.Sigmoid()  
    
    def forward(self, x):

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(out), dim=1) 
        context = torch.sum(out * attn_weights, dim=1)  
        
        # Final prediction
        out = self.dropout(context)
        out = self.fc(out)
        return self.sigmoid(out) 
def prepare_lstm_data(review_count, rating, vote, pop_score, seq_len=3):
    X, y = [], []
    scaler = MinMaxScaler()
    for item in review_count.index:
        item_data = np.vstack([pop_score.loc[item], review_count.loc[item], rating.loc[item], vote.loc[item]]).T
        if len(item_data) < seq_len + 1:
            continue
        scaled_data = scaler.fit_transform(item_data)
        for i in range(len(scaled_data) - seq_len):
            X.append(scaled_data[i:i + seq_len])
            y.append(scaled_data[i + seq_len, 0])  
    return np.array(X), np.array(y), scaler



def train_lstm(X_train, y_train, X_val, y_val, hidden_size=128, num_layers=3, epochs=50, patience=7, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedLSTMModel(input_size=4, hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(-1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(-1))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0) 
        train_loss /= len(train_dataset) 
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_dataset)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_model_state)
    return model

def time_series_forecast(review_count_wide_first, rating_wide_first, vote_wide_first, pop_score_wide_first, train_items, test_items, seq_len=12):
    X_all, y_all, scaler = prepare_lstm_data(
        review_count_wide_first.loc[train_items], rating_wide_first.loc[train_items],
        vote_wide_first.loc[train_items], pop_score_wide_first.loc[train_items], seq_len
    )
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    model = train_lstm(X_train, y_train, X_val, y_val, hidden_size=128, num_layers=3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts_predictions = []
    for item in review_count_wide_first.index:
        try:
            item_data = np.vstack([
                pop_score_wide_first.loc[item], review_count_wide_first.loc[item],
                rating_wide_first.loc[item], vote_wide_first.loc[item]
            ]).T
            if len(item_data) < seq_len + 1:
                continue
            scaled_data = scaler.transform(item_data)
            last_seq = scaled_data[-seq_len:]
            with torch.no_grad():
                input_seq = torch.FloatTensor(last_seq).unsqueeze(0).to(device)
                forecast = model(input_seq).cpu().numpy()[0][0]
                forecast_df = pd.DataFrame([[forecast, 0, 0, 0]], columns=['pop_score', 'review_count', 'rating', 'vote'])
                ts_predictions.append({'itemName': item, 'ts_prediction': scaler.inverse_transform(forecast_df)[0][0]})
        except KeyError:
            continue
    
    ts_df = pd.DataFrame(ts_predictions)
    return ts_df

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20, dropout=0.1):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, user, item):
        user_emb = self.dropout(self.user_factors(user))
        item_emb = self.dropout(self.item_factors(item))
        return (user_emb * item_emb).sum(1)

def tune_cf_parameters(df, n_users, n_items):
    best_score = float('inf')
    best_params = {}
    
    ratings = torch.FloatTensor(df['rating'].values).to(device)
    users = torch.LongTensor(df['user_id'].values).to(device)
    items = torch.LongTensor(df['item_id'].values).to(device)
    val_mask = np.random.rand(len(ratings)) < 0.2
    
    train_ratings = ratings[~val_mask]
    train_users = users[~val_mask]
    train_items = items[~val_mask]
    val_ratings = ratings[val_mask]
    val_users = users[val_mask]
    val_items = items[val_mask]
    
    for n_factors in [10, 20, 30]:
        for lr in [0.005, 0.01]:
            model = MatrixFactorization(n_users, n_items, n_factors=n_factors).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = nn.MSELoss()
            for epoch in range(20):
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(train_users, train_items), train_ratings)
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(val_users, val_items), val_ratings)
                    if val_loss < best_score:
                        best_score = val_loss
                        best_params = {'n_factors': n_factors, 'lr': lr}
    return best_params

def collaborative_filtering(df, items):
    df_train = df[df['itemName'].isin(items)].copy()
    user_map = {u: i for i, u in enumerate(df_train['userName'].unique())}
    item_map = {i: j for j, i in enumerate(df_train['itemName'].unique())}
    df_train['user_id'] = df_train['userName'].map(user_map)
    df_train['item_id'] = df_train['itemName'].map(item_map)
    
    class MFModel(torch.nn.Module):
        def __init__(self, num_users, num_items, embedding_dim=50):
            super(MFModel, self).__init__()
            self.user_emb = torch.nn.Embedding(num_users, embedding_dim)
            self.item_emb = torch.nn.Embedding(num_items, embedding_dim)
        
        def forward(self, users, items):
            user_vecs = self.user_emb(users)
            item_vecs = self.item_emb(items)
            return (user_vecs * item_vecs).sum(dim=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = MFModel(len(user_map), len(item_map)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    users = torch.LongTensor(df_train['user_id'].values).to(device)
    items_tensor = torch.LongTensor(df_train['item_id'].values).to(device)
    ratings = torch.FloatTensor(df_train['rating'].values).to(device)
    for _ in range(50):
        model.train()
        optimizer.zero_grad()
        preds = model(users, items_tensor)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()
    
    rec_counts = []
    for item in items:
        item_id = item_map.get(item, -1)
        if item_id == -1:
            rec_counts.append({'itemName': item, 'rec_count': 0})
        else:
            with torch.no_grad():
                preds = model(torch.arange(len(user_map)).to(device), torch.tensor([item_id]).repeat(len(user_map)).to(device))
                preds = torch.clamp(preds, min=1.0, max=5.0)
                # count predictions above 4.0 ("good" rating)
                rec_counts.append({'itemName': item, 'rec_count': (preds >= 4.0).sum().item()})
    cf_df = pd.DataFrame(rec_counts)
    return cf_df
def content_based_features(df, items):
    df_train = df[df['itemName'].isin(items)].drop_duplicates('itemName')
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_train['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    item_indices = {item: idx for idx, item in enumerate(df_train['itemName'])}
    
    sim_scores = []
    for item in items:
        if item in item_indices:
            idx = item_indices[item]
            sim_score = cosine_sim[idx].mean()
        else:
            sim_score = 0
        sim_scores.append({'itemName': item, 'content_sim': sim_score})
    sim_df = pd.DataFrame(sim_scores)
    return sim_df

# def tune_hybrid_model(X_train, y_train):
#     base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
#     param_grid = {
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.1],
#         'n_estimators': [100, 200],
#         'scale_pos_weight': [1, len(y_train[y_train == 0]) / len(y_train[y_train == 1])]
#     }
#     f1_scorer = make_scorer(f1_score)
#     grid_search = GridSearchCV(
#         estimator=base_model,
#         param_grid=param_grid,
#         scoring=f1_scorer,
#         cv=5,
#         n_jobs=-1,
#         verbose=1
#     )
#     grid_search.fit(X_train, y_train)
#     print(f"Best parameters: {grid_search.best_params_}")
#     print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
#     return grid_search.best_estimator_
def tune_hybrid_model(X_train, y_train):

    best_model = XGBClassifier(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=100,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        use_label_encoder=False,  
        eval_metric='logloss',    
        random_state=42           
    )
    
    best_model.fit(X_train, y_train)

    return best_model

def main_pipeline(df):
    time_split = '2018-06-01'
    df_pre_split = preprocess_data(df, time_split=time_split)
    
    (train_df, test_df, labels_df, train_items, test_items,
     review_count_wide_first, rating_wide_first, vote_wide_first, pop_score_wide_first, threshold) = create_popularity_labels(df_pre_split, time_split)
    df_first_part = df_pre_split[df_pre_split['reviewTime'] < pd.to_datetime(time_split)].copy()
    
    ts_df = time_series_forecast(review_count_wide_first, rating_wide_first, vote_wide_first, pop_score_wide_first, train_items, test_items)
    print('ts_df done')
    cf_df = collaborative_filtering(df_first_part, labels_df['itemName'])
    print('cf_df done')
    sim_df = content_based_features(df_first_part, labels_df['itemName'])
    print('sim_df done')
    
    final_df_train = labels_df[labels_df['itemName'].isin(train_items)].merge(ts_df, on='itemName', how='left')\
                                                                     .merge(cf_df, on='itemName', how='left')\
                                                                     .merge(sim_df, on='itemName', how='left')\
                                                                     .fillna(0)
    final_df_test = labels_df[labels_df['itemName'].isin(test_items)].merge(ts_df, on='itemName', how='left')\
                                                                    .merge(cf_df, on='itemName', how='left')\
                                                                    .merge(sim_df, on='itemName', how='left')\
                                                                    .fillna(0)
    
    X_train = final_df_train[['ts_prediction', 'rec_count', 'content_sim']]
    y_train = final_df_train['popular']
    X_test = final_df_test[['ts_prediction', 'rec_count', 'content_sim']]
    y_test = final_df_test['popular']
    print('done')
    
    # Baseline
    for quan in np.arange(0.4, 0.91, 0.05):
        baseline_preds = (X_test['ts_prediction'] > X_train['ts_prediction'].quantile(quan)).astype(int)
        baseline_f1 = f1_score(y_test, baseline_preds)
        print(f"Baseline F1 (TS Threshold): {baseline_f1:.4f}")
        baseline_precision = precision_score(y_test, baseline_preds)
        print(f"baseline_precision  (TS Threshold): {baseline_precision :.4f}")
        baseline_recall = recall_score(y_test, baseline_preds)
        print(f"baseline_recall  (TS Threshold): {baseline_recall:.4f}")
        baseline_accuracy = accuracy_score(y_test, baseline_preds)
        print(f"baseline_accuracy (TS Threshold): {baseline_accuracy:.4f}")
    model = tune_hybrid_model(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.4, 0.96, 0.05)
    for thresh in thresholds:
        predictions = (probs > thresh).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        print(f"Threshold {thresh}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    predictions = model.predict_proba(X_test)[:, 1]
    final_df_test['prediction'] = (predictions > thresh).astype(int)
    return final_df_test, model

df = pd.read_csv('data/amazon_reviews.csv')
result, model = main_pipeline(df)

result.to_csv('results.csv', index=False)
print("Results saved to 'results.csv'")


