Customer Segmentation — Clustering Analysis
An unsupervised machine learning pipeline that segments wholesale customers using K-Means, Hierarchical Clustering, and DBSCAN, with automatic hyperparameter tuning and visual evaluation.

Dataset
The project uses the Wholesale Customers dataset, containing 440 records across 8 features:
FeatureTypeDescriptionChannelint64Sales channel (1 = Horeca, 2 = Retail)Regionint64Geographic region (1, 2, or 3)Freshint64Annual spending on fresh productsMilkint64Annual spending on milk productsGroceryint64Annual spending on grocery productsFrozenint64Annual spending on frozen productsDetergents_Paperint64Annual spending on detergents & paperDelicassenint64Annual spending on delicatessen products
Data health: No missing values, no duplicate rows.

Pipeline Overview
main.py
  ├── 1. Data Health Check
  ├── 2. Exploratory Analysis (Pair Plot)
  ├── 3. Dimensionality Reduction (PCA)
  ├── 4. Hyperparameter Tuning
  │     ├── Elbow Plot      → K-Means k
  │     ├── Dendrogram      → Hierarchical n_clusters
  │     └── K-Distance Graph → DBSCAN eps
  ├── 5. Clustering
  │     ├── K-Means
  │     ├── Hierarchical
  │     └── DBSCAN
  └── 6. Evaluation & Comparison

Results
PCA reduced the feature space to 2 components, explaining 61.12% of total variance.
All three algorithms were run with 5 clusters (where applicable):
AlgorithmSilhouette ↑Davies-Bouldin ↓Calinski-Harabasz ↑K-Means0.53200.6616446.58Hierarchical0.53810.6343406.59DBSCAN0.85520.101759.19

↑ higher is better  |  ↓ lower is better

DBSCAN achieved the best Silhouette and Davies-Bouldin scores, though it identified only 1 cluster with 1 noise point — indicating the data may not have strong density-based structure. Hierarchical Clustering offers the best balance across all three metrics for interpretable multi-cluster segmentation.

Setup & Usage
Requirements
bashpip install -r requirements.txt
Running the pipeline
bashpython main.py
Plots are saved to files automatically — no display window required.
Matplotlib backend note
This project uses the Agg backend to avoid Tcl/Tk GUI dependencies:
pythonimport matplotlib
matplotlib.use('Agg')  # Set before importing pyplot
import matplotlib.pyplot as plt
All plots are saved as .png files instead of being shown interactively.

Output Files
FileDescriptionpair_plot.pngPairwise feature relationshipsscree_plot.pngPCA variance explained per componentelbow_plot.pngK-Means inertia vs. kdendrogram.pngHierarchical clustering dendrogramkdistance_plot.pngDBSCAN k-distance graph (suggested eps: 5.1239)

## Team Members
- Person 1: Initialization (.gitignore & README)
- Person 2: Data Loader (Error Handling)
- Person 3: EDA (Summary stats & Heatmap)
- Person 4: Preprocessing (Scikit-Learn Pipeline)
- Person 5: Dimensionality Reduction (PCA)
- Person 6: Clustering (K-Means)
- Person 7: Clustering (Hierarchical)
- Person 8: Clustering (DBSCAN)
- Person 9: Evaluation (Metrics & Visuals)
