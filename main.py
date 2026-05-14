import src.initialization as init
import src.data_loader as loader
import src.eda as eda
import src.preprocessing as preprocessor
import src.dimensionality as dim
import src.clustering as cluster
import src.evaluation as eval

def main():
    # 0. Initialization
    init.setup_project_structure()

    # 1. Load Data (No hard-coded paths - passed as a variable)
    df = loader.load_data("data/wholesale_customers.csv")
    if df is None:
        print("Data could not be loaded. Please download wholesale_customers.csv into the data folder.")
        return

    # 1.5 EDA
    eda.perform_full_eda(df) 
    eda.plot_visuals(df)     

    # 2. Build and run the Pipeline
    pipeline = preprocessor.get_pipeline()
    X_scaled = pipeline.fit_transform(df)

    # 3. PCA
    print("Generating PCA Scree Plot (close the pop-up window to continue)...")
    dim.plot_scree_plot(X_scaled)
    X_pca, variance = dim.apply_pca(X_scaled)
    print(f"PCA Explained Variance: {variance}")

    # 4. Run Algorithms (e.g., K-means)
    print("Generating Elbow Plot (close the pop-up window to continue)...")
    eval.plot_elbow_method(X_pca, max_k=10)
    km_labels, km_model = cluster.run_kmeans(X_pca, k=5)

    # 5. Evaluate
    score = eval.get_silhouette(X_pca, km_labels)
    print(f"K-means Silhouette Score: {score}")

    print("Opening PCA Clusters Graph (close the pop-up window to finish)...")
    eval.plot_pca_clusters(X_pca, km_labels, title="K-Means Clusters (k=5)")

if __name__ == "__main__":
    main()
