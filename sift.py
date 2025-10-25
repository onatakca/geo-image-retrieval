import cv2
import numpy as np
import json
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def extract_features(image_paths, n_features=500):
    detector = cv2.SIFT_create(nfeatures=n_features)

    all_descriptors = []
    
    for path in tqdm(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, descriptors = detector.detectAndCompute(img, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)
    
    return np.vstack(all_descriptors)


def build_vocabulary(descriptors, n_clusters=2000):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    kmeans.fit(descriptors)
    return kmeans


def create_bow_vectors(image_paths, kmeans, n_features=500, idf_weights=None):
    detector = cv2.SIFT_create(nfeatures=n_features)
    bow_vectors = []
    
    for path in tqdm(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            bow_vectors.append(np.zeros(kmeans.n_clusters))
            continue
            
        _, descriptors = detector.detectAndCompute(img, None)
        
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


def evaluate(query_vectors, db_vectors, ground_truth):
    recalls = {1: 0, 5: 0, 10: 0, 20: 0}
    average_precisions = []
    
    for q_idx in range(len(query_vectors)):
        similarities = np.dot(db_vectors, query_vectors[q_idx])
        top_indices = np.argsort(similarities)[::-1]
        relevant = set(ground_truth.get(q_idx, []))
        
        for k in [1, 5, 10, 20]:
            retrieved = set(top_indices[:k])
            if len(relevant & retrieved) > 0:
                recalls[k] += 1
        
        if len(relevant) > 0:
            precisions = []
            num_relevant_found = 0
            
            for i, db_idx in enumerate(top_indices[:20]):
                if db_idx in relevant:
                    num_relevant_found += 1
                    precision_at_i = num_relevant_found / (i + 1)
                    precisions.append(precision_at_i)
            
            if len(precisions) > 0:
                average_precisions.append(np.mean(precisions))
    
    for k in recalls:
        recalls[k] = recalls[k] / len(query_vectors)
    
    mAP = np.mean(average_precisions) if len(average_precisions) > 0 else 0
    return recalls, mAP


def main():
    N_FEATURES = 500
    VOCAB_SIZE = 2000
    
    with open('data/query/query_lite.json', 'r') as f:
        query_data = json.load(f)
    with open('data/database/database_lite.json', 'r') as f:
        db_data = json.load(f)
    
    query_paths = ['data/' + p for p in query_data['im_paths']]
    db_paths = ['data/' + p for p in db_data['im_paths']]
    
    descriptors = extract_features(db_paths, n_features=N_FEATURES)
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
    
    recalls, mAP = evaluate(query_vectors, db_vectors, ground_truth)
    
    print("for SIFT")
    for k, recall in recalls.items():
        print(f"recall is {k}: {recall:.4f} ({recall*100:.2f}%)")
    print(f"mAP: {mAP:.4f} ({mAP*100:.2f}%)")

if __name__ == "__main__":
    main()
