import cv2
import numpy as np
import json
import os
import pandas as pd
import pickle
import gc
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from scipy.spatial.distance import cdist

def extract_features(image_paths, n_features=500, batch_size=100):
    
    detector = cv2.SIFT_create(nfeatures=n_features)
    all_descriptors = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_descriptors = []

        for path in batch_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, descriptors = detector.detectAndCompute(img, None)
            if descriptors is not None:
                batch_descriptors.append(descriptors.astype(np.float32))

        if batch_descriptors:
            all_descriptors.append(np.vstack(batch_descriptors))

        del batch_descriptors
        gc.collect()

    return np.vstack(all_descriptors)

class SimpleFuzzyKMeans:
    

    def __init__(self, n_clusters, m=2.0, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices].copy()

        for _ in range(self.max_iter):
            distances = cdist(X, self.cluster_centers_, metric='euclidean')
            distances = np.fmax(distances, 1e-10)

            power = 2.0 / (self.m - 1)
            inv_distances = 1.0 / distances
            membership = inv_distances ** power
            membership = membership / np.sum(membership, axis=1, keepdims=True)

            membership_weighted = membership ** self.m
            new_centers = np.dot(membership_weighted.T, X) / np.sum(membership_weighted, axis=0)[:, np.newaxis]

            if np.allclose(self.cluster_centers_, new_centers, rtol=1e-4):
                break

            self.cluster_centers_ = new_centers

        return self

    def predict(self, X):
        distances = cdist(X, self.cluster_centers_, metric='euclidean')
        distances = np.fmax(distances, 1e-10)

        power = 2.0 / (self.m - 1)
        inv_distances = 1.0 / distances
        membership = inv_distances ** power
        membership = membership / np.sum(membership, axis=1, keepdims=True)

        return membership

def build_vocabulary(descriptors, n_clusters=2000, fuzzy=False, cache_path=None):
    
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached vocabulary from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    if fuzzy:
        print("Training Fuzzy K-means (may take longer)...")
        kmeans = SimpleFuzzyKMeans(n_clusters=n_clusters, random_state=42, max_iter=50)
    else:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)

    kmeans.fit(descriptors)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Caching vocabulary to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(kmeans, f)

    return kmeans

def create_bow_vectors_hard(image_paths, kmeans, n_features=500, idf_weights=None):
    
    detector = cv2.SIFT_create(nfeatures=n_features)
    bow_vectors = []

    for path in tqdm(image_paths, desc="Creating BoW (hard)"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            bow_vectors.append(np.zeros(kmeans.n_clusters, dtype=np.float32))
            continue

        _, descriptors = detector.detectAndCompute(img, None)
        if descriptors is None:
            bow_vectors.append(np.zeros(kmeans.n_clusters, dtype=np.float32))
            continue

        words = kmeans.predict(descriptors)
        histogram = np.zeros(kmeans.n_clusters, dtype=np.float32)
        for word in words:
            histogram[word] += 1

        if idf_weights is not None:
            histogram = histogram * idf_weights

        if np.linalg.norm(histogram) > 0:
            histogram = histogram / np.linalg.norm(histogram)

        bow_vectors.append(histogram)

    return np.array(bow_vectors, dtype=np.float32)

def create_bow_vectors_soft(image_paths, kmeans, n_features=500, idf_weights=None, top_k=3):
    
    detector = cv2.SIFT_create(nfeatures=n_features)
    bow_vectors = []

    for path in tqdm(image_paths, desc="Creating BoW (soft)"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            bow_vectors.append(np.zeros(kmeans.n_clusters, dtype=np.float32))
            continue

        _, descriptors = detector.detectAndCompute(img, None)
        if descriptors is None:
            bow_vectors.append(np.zeros(kmeans.n_clusters, dtype=np.float32))
            continue

        distances = cdist(descriptors.astype(np.float32), kmeans.cluster_centers_.astype(np.float32), metric='euclidean')

        histogram = np.zeros(kmeans.n_clusters, dtype=np.float32)
        for dist_row in distances:
            top_k_indices = np.argsort(dist_row)[:top_k]
            top_k_dists = dist_row[top_k_indices]
            weights = 1.0 / (top_k_dists + 1e-6)
            weights = weights / np.sum(weights)

            for idx, weight in zip(top_k_indices, weights):
                histogram[idx] += weight

        if idf_weights is not None:
            histogram = histogram * idf_weights

        if np.linalg.norm(histogram) > 0:
            histogram = histogram / np.linalg.norm(histogram)

        bow_vectors.append(histogram)

    return np.array(bow_vectors, dtype=np.float32)

def create_bow_vectors_fuzzy(image_paths, kmeans, n_features=500, idf_weights=None):
    
    detector = cv2.SIFT_create(nfeatures=n_features)
    bow_vectors = []

    for path in tqdm(image_paths, desc="Creating BoW (fuzzy)"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            bow_vectors.append(np.zeros(kmeans.n_clusters, dtype=np.float32))
            continue

        _, descriptors = detector.detectAndCompute(img, None)
        if descriptors is None:
            bow_vectors.append(np.zeros(kmeans.n_clusters, dtype=np.float32))
            continue

        memberships = kmeans.predict(descriptors.astype(np.float32))
        histogram = np.sum(memberships, axis=0).astype(np.float32)

        if idf_weights is not None:
            histogram = histogram * idf_weights

        if np.linalg.norm(histogram) > 0:
            histogram = histogram / np.linalg.norm(histogram)

        bow_vectors.append(histogram)

    return np.array(bow_vectors, dtype=np.float32)

def compute_idf(bow_vectors):
    
    N = len(bow_vectors)
    df = np.sum(bow_vectors > 0, axis=0)
    idf = np.log(N / (df + 1))
    return idf

def compute_similarity(query_vectors, db_vectors, metric='l2'):
    
    if metric == 'l2' or metric == 'cosine':
        similarities = np.dot(query_vectors, db_vectors.T)
    elif metric == 'l1':
        similarities = -cdist(query_vectors, db_vectors, metric='cityblock')
    elif metric == 'chi-square':
        similarities = np.zeros((len(query_vectors), len(db_vectors)))
        for i, q_vec in enumerate(query_vectors):
            for j, db_vec in enumerate(db_vectors):
                chi_sq = np.sum((q_vec - db_vec) ** 2 / (q_vec + db_vec + 1e-10))
                similarities[i, j] = -chi_sq
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return similarities

def recall(ranks, pidx, ks):
    
    recall_at_k = np.zeros(len(ks))
    for qidx in range(ranks.shape[0]):
        for i, k in enumerate(ks):
            if np.sum(np.isin(ranks[qidx, :k], pidx[qidx])) > 0:
                recall_at_k[i:] += 1
                break

    recall_at_k /= ranks.shape[0]
    return recall_at_k

def apk(pidx, rank, k):
    
    if len(rank) > k:
        rank = rank[:k]

    if len(pidx) == 0 or k == 0:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(rank):
        if p in pidx and p not in rank[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(pidx), k)

def mapk(ranks, pidxs, k):
    return np.mean([apk(a, p, k) for a, p in zip(pidxs, ranks)])

def mapk_many(ranks, pidxs, ks):
    
    return np.array([mapk(ranks, pidxs, k) for k in ks], dtype=float)

def evaluate(query_vectors, db_vectors, ground_truth, metric='l2'):
    
    similarities = compute_similarity(query_vectors, db_vectors, metric)
    ranks = np.argsort(-similarities, axis=1)

    Q = similarities.shape[0]
    pidx = [np.array(ground_truth.get(q, []), dtype=int) for q in range(Q)]

    ks = [1, 5, 10, 20]
    recall_at_k = recall(ranks, pidx, ks)
    mAPs = mapk_many(ranks, pidx, ks)

    recalls = {k: r for k, r in zip(ks, recall_at_k)}
    return recalls, mAPs

def run_experiment(config, query_data, db_data, query_paths, db_paths, ground_truth):
    print(f"Running: {config['name']}")
    print(f"Config: n_features={config['n_features']}, n_clusters={config['n_clusters']}, "
          f"assignment={config['assignment']}, distance={config['distance']}, tfidf={config['use_tfidf']}")

    cache_path = f"results/sift/vocab_cache/vocab_f{config['n_features']}_c{config['n_clusters']}_{'fuzzy' if config['assignment'] == 'fuzzy' else 'hard'}.pkl"

    descriptors = extract_features(db_paths, n_features=config['n_features'])
    fuzzy = (config['assignment'] == 'fuzzy')
    kmeans = build_vocabulary(descriptors, n_clusters=config['n_clusters'], fuzzy=fuzzy, cache_path=cache_path)

    del descriptors
    gc.collect()

    if config['assignment'] == 'hard':
        db_vectors = create_bow_vectors_hard(db_paths, kmeans, n_features=config['n_features'])
        query_vectors = create_bow_vectors_hard(query_paths, kmeans, n_features=config['n_features'])
    elif config['assignment'] == 'soft':
        db_vectors = create_bow_vectors_soft(db_paths, kmeans, n_features=config['n_features'])
        query_vectors = create_bow_vectors_soft(query_paths, kmeans, n_features=config['n_features'])
    elif config['assignment'] == 'fuzzy':
        db_vectors = create_bow_vectors_fuzzy(db_paths, kmeans, n_features=config['n_features'])
        query_vectors = create_bow_vectors_fuzzy(query_paths, kmeans, n_features=config['n_features'])

    if config['use_tfidf']:
        idf_weights = compute_idf(db_vectors)
        db_vectors_tfidf = db_vectors * idf_weights
        query_vectors_tfidf = query_vectors * idf_weights

        db_norms = np.linalg.norm(db_vectors_tfidf, axis=1, keepdims=True)
        query_norms = np.linalg.norm(query_vectors_tfidf, axis=1, keepdims=True)
        db_vectors = np.divide(db_vectors_tfidf, db_norms, where=db_norms > 0, out=db_vectors_tfidf)
        query_vectors = np.divide(query_vectors_tfidf, query_norms, where=query_norms > 0, out=query_vectors_tfidf)

    recalls, mAPs = evaluate(query_vectors, db_vectors, ground_truth, metric=config['distance'])

    del db_vectors, query_vectors
    gc.collect()

    ks = [1, 5, 10, 20]
    for k, recall_val, map_val in zip(ks, [recalls[k] for k in ks], mAPs):
        print(f"Recall@{k}: {recall_val*100:.2f}%   mAP@{k}: {map_val*100:.2f}%")

    result = {
        "experiment_name": config['name'],
        "n_features": config['n_features'],
        "n_clusters": config['n_clusters'],
        "assignment": config['assignment'],
        "distance": config['distance'],
        "use_tfidf": config['use_tfidf'],
        "Recall@1": recalls[1] * 100,
        "Recall@5": recalls[5] * 100,
        "Recall@10": recalls[10] * 100,
        "Recall@20": recalls[20] * 100,
        "mAP@1": mAPs[0] * 100,
        "mAP@5": mAPs[1] * 100,
        "mAP@10": mAPs[2] * 100,
        "mAP@20": mAPs[3] * 100
    }

    return result

def main():
    with open('data/query/query_lite.json', 'r') as f:
        query_data = json.load(f)
    with open('data/database/database_lite.json', 'r') as f:
        db_data = json.load(f)

    query_paths = ['data/' + p for p in query_data['im_paths']]
    db_paths = ['data/' + p for p in db_data['im_paths']]

    ground_truth = {}
    for q_idx, q_loc in enumerate(query_data['loc']):
        relevant = []
        for db_idx, db_loc in enumerate(db_data['loc']):
            distance = np.sqrt((q_loc[0] - db_loc[0])**2 + (q_loc[1] - db_loc[1])**2)
            if distance <= 25:
                relevant.append(db_idx)
        ground_truth[q_idx] = relevant

    csv_path = "results/sift/sift_ablation_results.csv"
    df_grid = pd.read_csv(csv_path)

    print(df_grid[['experiment_name', 'Recall@1', 'Recall@20', 'mAP@1']].to_string(index=False))

    best_idx = df_grid['Recall@1'].idxmax()
    best_n_feat = int(df_grid.loc[best_idx, 'n_features'])
    best_n_clust = int(df_grid.loc[best_idx, 'n_clusters'])

    print(f"Best configuration: n_features={best_n_feat}, n_clusters={best_n_clust}")
    print(f"Recall@1: {df_grid.loc[best_idx, 'Recall@1']:.2f}%")

    ablation_configs = [
        {
            'name': 'ablation_soft_assignment',
            'n_features': best_n_feat,
            'n_clusters': best_n_clust,
            'assignment': 'soft',
            'distance': 'l2',
            'use_tfidf': True
        },
        {
            'name': 'ablation_l1_distance',
            'n_features': best_n_feat,
            'n_clusters': best_n_clust,
            'assignment': 'hard',
            'distance': 'l1',
            'use_tfidf': True
        },
        {
            'name': 'ablation_cosine_distance',
            'n_features': best_n_feat,
            'n_clusters': best_n_clust,
            'assignment': 'hard',
            'distance': 'cosine',
            'use_tfidf': True
        },
        {
            'name': 'ablation_chisquare_distance',
            'n_features': best_n_feat,
            'n_clusters': best_n_clust,
            'assignment': 'hard',
            'distance': 'chi-square',
            'use_tfidf': True
        },
        {
            'name': 'ablation_no_tfidf',
            'n_features': best_n_feat,
            'n_clusters': best_n_clust,
            'assignment': 'hard',
            'distance': 'l2',
            'use_tfidf': False
        }
    ]

    results = df_grid.to_dict('records')

    for config in ablation_configs:
        result = run_experiment(config, query_data, db_data, query_paths, db_paths, ground_truth)
        results.append(result)

        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"Saved intermediate results to {csv_path}")

    df_final = pd.DataFrame(results)
    print("\nTop 5 configurations by Recall@1:")
    print(df_final.nlargest(5, 'Recall@1')[['experiment_name', 'Recall@1', 'Recall@20', 'mAP@1']].to_string(index=False))

if __name__ == "__main__":
    main()
