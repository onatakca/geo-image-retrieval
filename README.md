# Place Recognition Project Plan

## Dataset

Dataset used is Mapillary Street-level Sequences (https://github.com/mapillary/mapillary_sls) consisting of multiple towns in the world. We are given London as a subset, but we can expand or evaluate on any other dataset if needed.

## Task

Task is that of place recognition: we have 1000 gallery images and 500 query images. Gallery images are those that are in our database and that we need to extract when fed an image from a query. Annotations/labels in this case are binary, 1 if collection image is relevant to an image and 0 otherwise, with relevance meaning the image represents the same place (street) in our dataset.

## Baseline Implementation (Assignment 2)

In the second assignment, we are trying out BoW approach as a baseline. For each image extract 50 ORB keypoints, create a matrix of N_of_query_images x N_of_collection_images and we're using K-means to cluster them, then we're using centroids to create histograms for each image and then based on it we're creating vectors for all images, which are then compared with similarity metric to find the relevant images for a query.

- [DONE] BASELINE: BoW using ORB features

## Upgrades

Upgrades can be (logically ordered as improvements):

- [ ] BoW using SIFT features
- [ ] BoW using SURF features
- [ ] Similarity search using features from DINOV3 backbone
- [ ] Similarity search using features from CLIP ReID backbone
- [ ] [OPTIONAL] Try using both or logistic regression over features + SURF/SIFT
- [ ] [DECIDE] if CLIP or DINOV3 performs better and then
- [ ] Do location grouping (we have GPS data) and for each location for all images extract features using better model, average them and use them as a location representation
  - [ ] Then, use those centroids for similarity search
    - [ ] Approach a) first decide which location is the most similar then search for relevant images there
    - [ ] Approach b) use metric: `alfa*similarity_to_centroid + (1-alfa)*similarity_to_image` to find relevant images
- [ ] All this assumed no knowledge of location (possible in inference)
- [ ] We can use GPS data then to drop the centroids and use GPS's to filter relevant images and then do retrieval, comparison to the previous approaches

## Evaluation Metrics

- [ ] Recall @1,5,10,20 (official metrics)
- [ ] mAP
- [ ] Latency & memory (to see if centroids approach is faster)
