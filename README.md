# Foundational Models for Zero-Shot Image Retrieval  
**Evaluation on the MSLS London Subset**

## Overview

This study examines how **foundation models** can perform **visual place recognition** without additional training, using the **Mapillary Street-Level Sequences (MSLS)** dataset. We focus on the **London** subset, where the task is to match query images to their corresponding locations in a gallery of geotagged photos.

- **Gallery:** 1,000 images with GPS coordinates  
- **Queries:** 500 images without location data
Our objective is to identify the most visually/spatially relevant matches for each query. To do so, we explore both **traditional BoW approaches** with **zero-shot probing of vision backbones** of recent foundation models.
---

## Experimental Focus

We explore several retrieval paradigms:

1. **Bag-of-Words (BoW)** models using handcrafted local descriptors — **ORB**, **SURF**, and **SIFT** — combined with clustering and TF-IDF weighting.  
2. **Zero-shot linear probing** of pretrained backbones — **DINOv3**, **Perception Encoder**, **CLIP**, and **StreetCLIP** — using different **feature pooling** strategies (CLS, mean, mean without CLS token, max, and GeM).  We ablte pooling strategy performances.
3. **GPS-aided search**, where gallery features are grouped by their geolocation and represented by centroid embeddings.  
4. **Hybrid backbones**, where features from multiple models (DINOv3-H/16+ and CLIP-B/32) are concatenated to explore combined prformance.  

Evaluation relies on **Recall@K** (K = 1, 5, 10, 20) and **mAP@K**.

---

## Dataset

- **Source:** [Mapillary Street-Level Sequences (MSLS)](https://github.com/mapillary/mapillary_sls)  
- **Subset:** London  
- **Annotations:** Each query–gallery pair is labeled as relevant (1) or non-relevant (0) based on geographic overlap.

---

## Task Definition

For each query image, retrieve the gallery image(s) that depict the same location or street segment.  
Success is measured by how often the correct location appears in the top-K retrieved results.

---

## Baseline: Bag-of-Words (BoW)

The classical BoW pipeline serves as a historical baseline:

1. Extract local descriptors (ORB, SURF, or SIFT).  
2. Learn a **visual vocabulary** via K-means clustering.  
3. Represent each image as a **TF-IDF weighted histogram** over visual words.  
4. Compare images using distance metrics such as L2 or cosine similarity.

### Baseline Results (Recall %)

| Method | R@1 | R@5 | R@10 | R@20 |
|---------|----:|----:|-----:|-----:|
| ORB | 1.63 | 3.71 | 5.35 | 9.21 |
| SURF + TF-IDF | 2.90 | 8.17 | 11.52 | 18.39 |
| **SIFT + TF-IDF** | **8.80** | **17.42** | **22.25** | **28.83** |

### SIFT Variants (Recall %)

| Configuration | R@1 | R@5 | R@10 | R@20 |
|---------------|----:|----:|-----:|-----:|
| n=500, k=2000 (hard, L2, TF-IDF) | 12.2 | 29.0 | 41.0 | 48.2 |
| **Soft assignment** | **13.2** | **31.2** | **40.2** | **49.2** |
| Without TF-IDF | 12.0 | 30.4 | 38.0 | 47.0 |

> **Best configuration:** SIFT + soft assignment + L2 distance + TF-IDF → **13.2 % Recall@1**

---

## Zero-Shot Linear Probing

We benchmark modern visual backbones in a **zero-shot** setup—no fine-tuning, just feature extraction and similarity search.

- **Models tested:** DINOv3 (ViT and ConvNeXt variants), Perception Encoder, CLIP, StreetCLIP  
- **Similarity metric:** cosine distance  
- **Pooling:** CLS, mean, mean-noCLS, max, and GeM  

### Pooling Comparison (Δ over CLS baseline)

| Pooling | ΔR@1 | ΔR@5 | ΔR@10 | ΔR@20 |
|----------|-----:|-----:|------:|------:|
| **GeM** | **+5.13** | **+7.51** | **+7.93** | **+7.47** |
| Max | +4.20 | +6.64 | +6.64 | +6.24 |
| Mean | +3.13 | +5.31 | +5.50 | +5.64 |
| Mean-noCLS | +3.10 | +5.19 | +5.51 | +5.51 |
| CLS | 0.00 | 0.00 | 0.00 | 0.00 |

**Conclusion:** **GeM pooling** provides the best retrieval performance across tested architectures.

---

## Linear Probing Results (Recall %)

### Perception Encoder
| Model | R@1 | R@5 | R@10 | R@20 |
|--------|----:|----:|-----:|-----:|
| PE-T16-384 | 24.4 | 45.4 | 58.6 | 71.0 |
| PE-S16-384 | 24.8 | 43.6 | 57.2 | 70.0 |
| PE-B16-224 | 28.8 | 57.8 | 69.2 | 79.6 |
| PE-L14-336 | 31.0 | 57.2 | 65.6 | 76.2 |
| PE-G14-448 | 26.2 | 50.2 | 61.4 | 72.2 |

---

### CLIP
| Model | R@1 | R@5 | R@10 | R@20 |
|--------|----:|----:|-----:|-----:|
| CLIP-B/32 (GeM) | 38.8 | 61.4 | 72.8 | 82.2 |
| CLIP-B/16 (Mean) | 31.0 | 51.8 | 64.0 | 76.8 |
| CLIP-L/14 (Mean-noCLS) | 32.6 | 55.2 | 67.0 | 80.4 |
| StreetCLIP (GeM) | 32.6 | 55.8 | 66.0 | 74.6 |

---

### DINOv3 (ViT)
| Model | R@1 | R@5 | R@10 | R@20 |
|--------|----:|----:|-----:|-----:|
| ViT-S/16 (CLS) | 38.0 | 57.0 | 65.6 | 77.6 |
| ViT-S/16+ (CLS) | 33.6 | 52.8 | 64.4 | 73.6 |
| ViT-B/16 (GeM) | 41.4 | 63.6 | 73.8 | **85.4** |
| ViT-L/16 (Max) | 40.4 | 59.0 | 69.8 | 80.2 |
| **ViT-H/16+ (GeM)** | **44.0** | **64.8** | **73.4** | 82.4 |

---

### DINOv3 (ConvNeXt)
| Model | R@1 | R@5 | R@10 | R@20 |
|--------|----:|----:|-----:|-----:|
| CNX-T (Max) | 26.6 | 45.6 | 55.0 | 67.2 |
| CNX-S (GeM) | 20.6 | 41.0 | 52.8 | 60.8 |
| CNX-B (GeM) | 24.6 | 37.2 | 45.4 | 55.6 |
| CNX-L (Max) | 24.2 | 47.4 | 55.4 | 64.2 |

---

### Multi-Backbone Combination
| Combination | R@1 | R@5 | R@10 | R@20 |
|--------------|----:|----:|-----:|-----:|
| **DINOv3-H/16+ + CLIP-B/32** | **44.8** | **66.6** | **76.0** | 85.0 |

---

## GPS-Aided Retrieval

To exploit spatial context, gallery features were clustered based on their GPS coordinates, creating centroid embeddings representing approximate locations. During retrieval, the similarity score blends direct image similarity with centroid proximity.
Formula we used here was:

<img width="279" height="36" alt="image" src="https://github.com/user-attachments/assets/98b859df-2bd9-4274-aefe-3a0504c35fd7" />

where q is the query image, I_i is a gallery image, c_j is the matched centroid, and alpha ∈ [0,1] is a weighting parameter.


### DINOv3-H/16+ (GeM)
| Method | R@1 | R@5 | R@10 | R@20 |
|---------|----:|----:|-----:|-----:|
| Baseline (no GPS) | 44.0 | 64.8 | 73.4 | **82.4** |
| Cluster-first search | 43.2 | 61.6 | 70.0 | 77.2 |
| **Weighted (α = 0.5)** | **44.4** | **65.6** | **75.6** | 81.8 |

### DINOv3-H/16+ + CLIP-B/32
| Method | R@1 | R@5 | R@10 | R@20 |
|---------|----:|----:|-----:|-----:|
| Baseline (no GPS) | 44.8 | 66.6 | 76.0 | **85.0** |
| **Weighted (α = 0.6)** | **46.4** | **66.6** | **76.2** | 81.8 |

> **Best performance:** 46.4 % Recall@1 with GPS-weighted hybrid features.

---

## Conclusions

- The **SIFT-based BoW** approach reaches only **13.2 % Recall@1**.  
- **DINOv3-H/16+ (GeM pooling)** yields  **44 % Recall@1** out of the box.  
- **Combining DINOv3 and CLIP** features adds a modest improvement (44.8 %).  
- Incorporating **GPS priors** further refines retrieval performance with **46.4 % Recall@1**.  
- Across all architectures, **GeM pooling** consistently offers the most reliable embeddings.

---

## References

- Warburg et al., *Mapillary Street-Level Sequences (MSLS)*, CVPR 2020  
- Simeoni et al., *DINOv3*, 2025  
- Bolya et al., *Perception Encoder*, 2025  
- Radford et al., *CLIP*, 2021  
- Haas et al., *StreetCLIP*, 2023  
- Lowe, *SIFT*, 2004; Bay et al., *SURF*, 2006; Rublee et al., *ORB*, 2011  
- Wieczorek et al., *Centroid-based retrieval*, 2021  

---
