# Unsupervised Learning Capstone — Wholesale Customers

**ML II · Group 4 · UCI Wholesale Customers Dataset**

> A modular, reproducible clustering pipeline applying **K-Means**, **Hierarchical (Agglomerative)**, and **DBSCAN** to the UCI Wholesale Customers dataset, with **PCA** for dimensionality reduction and full internal metric evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Module Reference](#module-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Team Members](#team-members)
- [References](#references)

---

## Project Overview

This capstone applies four core unsupervised learning techniques to a real-world dataset:

| Technique | Purpose |
|---|---|
| **EDA** | Understand feature distributions, correlations, and outliers |
| **PCA** | Reduce dimensionality; retain ≥ 95% explained variance |
| **K-Means** | Partition-based clustering; tuned via elbow method |
| **Hierarchical** | Linkage-based clustering; visualised via dendrogram |
| **DBSCAN** | Density-based clustering; tuned via k-distance graph |

All three algorithms are evaluated on three mandatory internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and compared in a summary table.

---

## Repository Structure

```
ML-Capstone-Project-Group-4-WholesaleDataSet/
│
├── data/
│   ├── README.md                   ← Download instructions for the dataset
│   └── wholesale_customers.csv     ← (Downloaded Dataset)
│
├── notebooks/
│   └── Capstone Project Group 4.ipynb  ← Exploratory notebook (EDA & prototyping)
│
├── src/
│   ├── __init__.py
│   ├── initialization.py           ← Project structure setup
│   ├── data_loader.py              ← CSV loading with error handling
│   ├── eda.py                      ← EDA: stats, heatmap, boxplots, pairplot
│   ├── preprocessing.py            ← Sklearn Pipeline: imputer → scaler
│   ├── dimensionality.py           ← PCA + scree plot
│   ├── clustering.py               ← K-Means, Hierarchical, DBSCAN
│   └── evaluation.py               ← Metrics, elbow, k-distance, dendrogram, comparison table
│
├── main.py                         ← Entry point — runs the full pipeline
├── requirements.txt                ← All dependencies
├── .gitignore
└── LICENSE
```

> **Note:** `wholesale_customers.csv` is excluded from version control per `.gitignore`. See [`data/README.md`](data/README.md) for download instructions.

---

## Dataset

**UCI Wholesale Customers Dataset**
- **Instances:** 440
- **Features:** 8 numeric (annual spending in monetary units across product categories)
- **Ground Truth:** None — purely unsupervised
- **Best suited for:** Marketing segmentation, K-Means, DBSCAN

| Feature | Description |
|---|---|
| Fresh | Annual spending on fresh products |
| Milk | Annual spending on milk products |
| Grocery | Annual spending on grocery products |
| Frozen | Annual spending on frozen products |
| Detergents_Paper | Annual spending on detergents and paper |
| Delicatessen | Annual spending on delicatessen products |
| Channel | Sales channel (1 = HoReCa, 2 = Retail) |
| Region | Geographic region (1–3) |

📥 **Download:** [archive.ics.uci.edu/dataset/292](https://archive.ics.uci.edu/dataset/292/wholesale+customers+data)
Place the file at `data/wholesale_customers.csv`.

---

## Pipeline Architecture

```
Load Data (data_loader.py)
       ↓
EDA: Stats, Heatmap, Boxplots, Pairplot (eda.py)
       ↓
Preprocessing: Impute → StandardScale (preprocessing.py)
       ↓
PCA: Scree Plot → 2-Component Reduction (dimensionality.py)
       ↓
Hyperparameter Tuning:
  ├── Elbow Plot        → optimal k  (K-Means)
  ├── Dendrogram        → optimal n  (Hierarchical)
  └── K-Distance Graph  → optimal ε  (DBSCAN)
       ↓
Clustering (clustering.py):
  ├── K-Means
  ├── Hierarchical (Agglomerative, Ward)
  └── DBSCAN
       ↓
Evaluation (evaluation.py):
  ├── Silhouette Score
  ├── Davies-Bouldin Index
  ├── Calinski-Harabasz Index
  └── Comparative Summary Table
```

---

## Setup & Installation

> ⚠️ **Windows users:** Use a standard Python installation from [python.org](https://www.python.org/downloads/), not MSYS2 or minimal environments.

### 1. Clone the repository

```bash
git clone https://github.com/Phillybewillin/ML-Capstone-Project-Group-4-WholesaleDataSet.git
cd ML-Capstone-Project-Group-4-WholesaleDataSet
```

### 2. Create and activate a virtual environment

```bash
# Windows
py -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Follow the instructions in [`data/README.md`](data/README.md) and place `wholesale_customers.csv` inside the `data/` folder.

---

## Usage

Run the full end-to-end pipeline:

```bash
python main.py
```

The pipeline will:
1. Run EDA and display all plots (close each window to advance)
2. Generate the PCA scree plot
3. Display hyperparameter tuning plots for all three algorithms
4. Run K-Means, Hierarchical, and DBSCAN
5. Print evaluation metrics and a comparison table to the terminal

To explore interactively, open the notebook:

```bash
jupyter notebook notebooks/Capstone\ Project\ Group\ 4.ipynb
```

---

## Module Reference

| Module | Function | Description |
|---|---|---|
| [`src/data_loader.py`](src/data_loader.py) | `load_data(path)` | Loads CSV with full error handling |
| [`src/eda.py`](src/eda.py) | `perform_full_eda(df)` | Prints health report: nulls, dupes, dtypes, stats |
| [`src/eda.py`](src/eda.py) | `plot_visuals(df)` | Heatmap, boxplots, pairplot |
| [`src/preprocessing.py`](src/preprocessing.py) | `get_pipeline()` | Returns `SimpleImputer → StandardScaler` pipeline |
| [`src/dimensionality.py`](src/dimensionality.py) | `apply_pca(data, n_components)` | Runs PCA, returns transformed data + variance ratios |
| [`src/dimensionality.py`](src/dimensionality.py) | `plot_scree_plot(data)` | Cumulative explained variance plot |
| [`src/clustering.py`](src/clustering.py) | `run_kmeans(data, k)` | K-Means clustering |
| [`src/clustering.py`](src/clustering.py) | `run_hierarchical(data, n_clusters)` | Agglomerative clustering (Ward linkage) |
| [`src/clustering.py`](src/clustering.py) | `run_dbscan(data, eps, min_samples)` | DBSCAN density clustering |
| [`src/evaluation.py`](src/evaluation.py) | `plot_elbow_method(data, max_k)` | Elbow plot for K-Means k selection |
| [`src/evaluation.py`](src/evaluation.py) | `plot_dendrogram(data)` | Dendrogram for hierarchical n selection |
| [`src/evaluation.py`](src/evaluation.py) | `plot_k_distance(data, k)` | K-distance graph for DBSCAN ε selection |
| [`src/evaluation.py`](src/evaluation.py) | `plot_pca_clusters(data, labels, title)` | PCA scatter coloured by cluster |
| [`src/evaluation.py`](src/evaluation.py) | `print_comparison_table(results)` | Prints full algorithm × metric comparison table |

---

## Evaluation Metrics

| Metric | Direction | What it measures |
|---|---|---|
| **Silhouette Score** | ↑ higher is better | How well-separated each point is from other clusters. Range: [-1, 1] |
| **Davies-Bouldin Index** | ↓ lower is better | Ratio of within-cluster scatter to between-cluster distance |
| **Calinski-Harabasz Index** | ↑ higher is better | Ratio of between-cluster to within-cluster dispersion |

> No ground truth labels are available for this dataset, so external metrics (ARI, NMI) are not applicable.

---

## Results

> Full visualisations, cluster plots, and comparative analysis are in [`report.pdf`](report.pdf).

### Algorithm Comparison Table

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| **K-Means** | 0.5320 | 0.6616 | **446.58** |
| **Hierarchical** | 0.5381 | 0.6343 | 406.59 |
| **DBSCAN** | **0.8552** | **0.1017** | 59.19 |

↑ higher is better &nbsp;|&nbsp; ↓ lower is better

### Key Observations

- **DBSCAN** achieves the best Silhouette (0.855) and Davies-Bouldin (0.102), indicating it finds the tightest, most well-separated density clusters. Its low Calinski-Harabasz (59.19) is expected — this metric favours many compact clusters, and DBSCAN produced fewer, larger ones.
- **Hierarchical** marginally outperforms K-Means on Silhouette and Davies-Bouldin, suggesting Ward linkage captures the cluster structure of this dataset slightly better than centroid-based partitioning.
- **K-Means** scores highest on Calinski-Harabasz (446.58), reflecting stronger between-cluster vs within-cluster dispersion across its 5 partitions — useful for marketing segmentation where distinct, balanced groups are preferred.
- No ground truth labels exist for this dataset, so external metrics (ARI, NMI) are not applicable.

---

## Team Members

| Member | Contribution |
|---|---|
| Denis | Repo setup, `initialization.py`, `.gitignore`, `README.md` |
| Vaz | `data_loader.py` — CSV loading & error handling |
| Tervin  | `eda.py` — Summary stats, heatmap, boxplots, pairplot |
| Ashad | `preprocessing.py` — Sklearn imputer → scaler pipeline |
| Mike | `dimensionality.py` — PCA + scree plot |
| Ian | `clustering.py` — K-Means, Hierarchical & DBSCAN implementation |
| Markphil | `Evaluation.py` — Metrics, elbow, k-distance, dendrogram, comparison table |

---

## References

- Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository](https://archive.ics.uci.edu). Irvine, CA: University of California, School of Information and Computer Science.
- Wholesale Customers Dataset: [archive.ics.uci.edu/dataset/292](https://archive.ics.uci.edu/dataset/292/wholesale+customers+data)
- [scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn Decomposition (PCA)](https://scikit-learn.org/stable/modules/decomposition.html#pca)
