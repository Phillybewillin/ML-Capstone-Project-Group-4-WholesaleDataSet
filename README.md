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
| **PCA** | Reduce dimensionality to 3 components — retaining 73.77% explained variance |
| **K-Means** | Partition-based clustering; tuned via elbow method |
| **Hierarchical** | Linkage-based clustering; visualised via dendrogram |
| **DBSCAN** | Density-based clustering; tuned via k-distance graph (ε=0.8) |

All three algorithms are evaluated on three mandatory internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and compared in a summary table.

---

## Repository Structure

```
ML-Capstone-Project-Group-4-WholesaleDataSet/
│
├── data/
│   ├── README.md                   ← Download instructions for the dataset
│   └── wholesale_customers.csv     ← (NOT committed — download manually)
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
- **Best suited for:** Marketing segmentation

| Feature | Description |
|---|---|
| Fresh | Annual spending on fresh products |
| Milk | Annual spending on milk products |
| Grocery | Annual spending on grocery products |
| Frozen | Annual spending on frozen products |
| Detergents_Paper | Annual spending on detergents and paper |
| Delicassen | Annual spending on delicatessen products |
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
PCA: Scree Plot → 3-Component Reduction — 73.77% variance (dimensionality.py)
       ↓
Hyperparameter Tuning:
  ├── Elbow Plot        → optimal k = 5  (K-Means)
  ├── Dendrogram        → optimal n = 5  (Hierarchical)
  └── K-Distance Graph  → optimal ε = 0.8  (DBSCAN)
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
2. Generate the PCA scree plot (3 components, 73.77% variance)
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
| **K-Means (k=5)** | **0.4577** ★ | 0.7661 | **270.75** ★ |
| **Hierarchical (k=5, Ward)** | 0.4410 | 0.7733 | 234.92 |
| **DBSCAN (ε=0.8, ms=5)** | 0.0489 | 1.3274 | 49.23 |

↑ higher is better &nbsp;|&nbsp; ↓ lower is better &nbsp;|&nbsp; ★ = best in column

### Key Observations

- **K-Means** achieves the best Silhouette (0.4577) and Calinski-Harabasz (270.75), producing the most well-separated and globally compact cluster structure. It is the recommended algorithm for marketing segmentation on this dataset, with directly interpretable centroids representing mean customer spending profiles.
- **Hierarchical** performs comparably to K-Means (Silhouette 0.4410, DBI 0.7733), confirming Ward linkage captures a similar cluster structure. The dendrogram provides additional exploratory value.
- **DBSCAN** scores poorly across all three metrics (Silhouette 0.0489, DBI 1.3274, CH 49.23). This is a meaningful finding - the Wholesale Customers dataset has a continuous, globular structure with no density-based separation, ruling out DBSCAN for this segmentation task. 6 clusters were identified with 32 noise points (7.3% of the dataset).
- No ground truth labels exist for this dataset, so external metrics (ARI, NMI) are not applicable.

---

## Team Members

| Member | Admission No. | Contribution |
|---|---|---|
| Dennis Momanyi | SCT213-C002-0082/2023 | Repo setup, `initialization.py`, `.gitignore`, `README.md` |
| Edwin Vaz Muiruri | SCT213-C002-0133/2023 | `data_loader.py` — CSV loading & error handling |
| Tervil Moywaywa | SCT213-C002-0012/2023 | `eda.py` — Summary stats, heatmap, boxplots, pairplot |
| Daniel Mburu | SCT213-C002-0008/2023 | `preprocessing.py` — Sklearn imputer → scaler pipeline |
| Michael Ahereza | SCT213-C002-0097/2022 | `dimensionality.py` — PCA + scree plot |
| Ian Nyaberi | SCT213-C002-0137/2023 | `clustering.py` — K-Means, Hierarchical & DBSCAN |
| Markphil Okao | SCT213-C002-0047/2023 | `evaluation.py` — Metrics, elbow, k-distance, dendrogram, table |
| Jeremiah Onywere | SCT213-C002-0097/2023 | Report compilation & submission |

---

## References

- Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository](https://archive.ics.uci.edu). Irvine, CA: University of California, School of Information and Computer Science.
- Wholesale Customers Dataset: [archive.ics.uci.edu/dataset/292](https://archive.ics.uci.edu/dataset/292/wholesale+customers+data)
- [scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn Decomposition (PCA)](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- Ester, M., et al. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. KDD-96.
