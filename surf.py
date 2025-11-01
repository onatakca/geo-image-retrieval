import cv2
import numpy as np
import json
import os
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from mahotas.features import surf
import mahotas as mh


def extract_surf_descriptors(image_paths, max_points=500, sigma=5.0):
    all_desc = []
    D = 128 

    for path in tqdm(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = img.astype(np.float32)
        if sigma:
            img = mh.gaussian_filter(img, sigma)

        desc = surf.surf(img, max_points=max_points, descriptor_only=True).astype(np.float32)
        if desc is None or desc.size == 0:
            continue

        norms = np.linalg.norm(desc, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        all_desc.append(desc / norms)

    return np.vstack(all_desc)

def build_vocabulary(descriptors, n_clusters=2000):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    kmeans.fit(descriptors)
    return kmeans


def create_bow_vectors(image_paths, kmeans, n_features=500, idf_weights=None):
    bow_vectors = []
    
    for path in tqdm(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            bow_vectors.append(np.zeros(kmeans.n_clusters))
            continue
            
        descriptors = surf.surf(img, max_points=n_features, descriptor_only=True).astype(np.float32)
        
        if descriptors is None:
            bow_vectors.append(np.zeros(kmeans.n_clusters))
            continue
        
        words = kmeans.predict(descriptors)
        
        histogram = np.zeros(kmeans.n_clusters)
        for word in words:
            histogram[word] += 1
        
        if idf_weights is not None:
            histogram = histogram * idf_weights
        
        if np.linalg.norm(histogram) > 0:
            histogram = histogram / np.linalg.norm(histogram)
        
        bow_vectors.append(histogram)
    
    return np.array(bow_vectors)


def compute_idf(bow_vectors):
    N = len(bow_vectors)
    df = np.sum(bow_vectors > 0, axis=0)
    idf = np.log(N / (df + 1))
    return idf


def recall(ranks, pidx, ks):
    recall_at_k = np.zeros(len(ks))
    for qidx in range(ranks.shape[0]):
        for i, k in enumerate(ks):
            if np.sum(np.isin(ranks[qidx,:k], pidx[qidx])) > 0:
                recall_at_k[i:] += 1
                break

    recall_at_k /= ranks.shape[0]
    return recall_at_k

def apk(pidx, rank, k):
    if len(rank)>k:
        rank = rank[:k]

    if len(pidx) == 0 or k == 0:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(rank):
        if p in pidx and p not in rank[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(pidx), k)

def mapk(ranks, pidxs, k):
    return np.mean([apk(a,p,k) for a,p in zip(pidxs, ranks)])

def mapk_many(ranks, pidxs, ks):
    return np.array([mapk(ranks, pidxs, k) for k in ks], dtype=float)

def evaluate(query_vectors, db_vectors, ground_truth):
    similarities = np.dot(query_vectors, db_vectors.T)

    ranks = np.argsort(-similarities, axis=1)

    Q = similarities.shape[0]
    pidx = [np.array(ground_truth.get(q, []), dtype=int) for q in range(Q)]

    ks = [1, 5, 10, 20]
    recall_at_k = recall(ranks, pidx, ks)
    mAPs = mapk_many(ranks, pidx, ks)

    recalls = {k: r for k, r in zip(ks, recall_at_k)}

    return recalls, mAPs


def main():
    N_FEATURES = 500
    VOCAB_SIZE = 2000
    
    with open('data/query/query.json', 'r') as f:
        query_data = json.load(f)
    with open('data/database/database.json', 'r') as f:
        db_data = json.load(f)
    
    query_paths = ['data/' + p for p in query_data['im_paths']]
    db_paths = ['data/' + p for p in db_data['im_paths']]
    
    descriptors = extract_surf_descriptors(db_paths, max_points=N_FEATURES)
    kmeans = build_vocabulary(descriptors, n_clusters=VOCAB_SIZE)
    
    db_vectors = create_bow_vectors(db_paths, kmeans, n_features=N_FEATURES)
    
    idf_weights = compute_idf(db_vectors)

    db_vectors = create_bow_vectors(db_paths, kmeans, n_features=N_FEATURES, idf_weights=idf_weights)
    query_vectors = create_bow_vectors(query_paths, kmeans, n_features=N_FEATURES, idf_weights=idf_weights)
    
    ground_truth = {}
    for q_idx, q_loc in enumerate(query_data['loc']):
        relevant = []
        for db_idx, db_loc in enumerate(db_data['loc']):
            distance = np.sqrt((q_loc[0] - db_loc[0])**2 + (q_loc[1] - db_loc[1])**2)
            if distance <= 25:
                relevant.append(db_idx)
        ground_truth[q_idx] = relevant
    
    recalls, mAPs = evaluate(query_vectors, db_vectors, ground_truth)

    print("for SURF")
    ks = [1, 5, 10, 20]
    for k, recall_val, map_val in zip(ks, [recalls[k] for k in ks], mAPs):
        print(f"Recall@{k}: {recall_val*100:.2f}%   mAP@{k}: {map_val*100:.2f}%")

    # Save results to CSV
    results_dict = {
        "models_name": "SURF_BoW_TF-IDF",
        "Recall@1": recalls[1] * 100,
        "Recall@5": recalls[5] * 100,
        "Recall@10": recalls[10] * 100,
        "Recall@20": recalls[20] * 100,
        "mAP@1": mAPs[0] * 100,
        "mAP@5": mAPs[1] * 100,
        "mAP@10": mAPs[2] * 100,
        "mAP@20": mAPs[3] * 100
    }

    csv_path = "results/surf/surf_results.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "surf_BoW_TF-IDF" in df['models_name'].values:
            df.loc[df['models_name'] == "surf_BoW_TF-IDF"] = pd.Series(results_dict)
        else:
            df = pd.concat([df, pd.DataFrame([results_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([results_dict])

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()
