"""
KMeans Clustering for Crime Pattern Analysis at State Level

This module performs OFFLINE clustering of Indian states based on 
aggregated crime statistics (2001-2014).

Purpose:
- Identify states with similar crime patterns
- Support area-wise crime analysis
- Enable policy-focused insights

Ethical Compliance:
- Uses only aggregated state-level data
- No individual profiling
- Academic research purpose only
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class CrimeClusteringModel:
    """
    State-level crime clustering using KMeans.
    
    Features used:
    - total_ipc_crimes (aggregated 2001-2014)
    - total_women_crimes (aggregated 2001-2014)
    - total_children_crimes (aggregated 2001-2014)
    """
    
    def __init__(self, data_path="data/processed"):
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.model = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.silhouette = None
        self.inertia = None
        
    def load_data(self):
        """
        Load preprocessed crime data for clustering.
        
        Returns:
            DataFrame with state-level aggregated crime statistics
        """
        # Load the three main datasets
        core_ipc = pd.read_csv(self.data_path / "core_crime" / "core_ipc_standardized.csv")
        women_crime = pd.read_csv(self.data_path / "women_crime" / "women_crime_standardized.csv")
        children_crime = pd.read_csv(self.data_path / "children_crime" / "children_crime_standardized.csv")
        
        # Aggregate by state
        state_ipc = (
            core_ipc
            .groupby("state_ut", as_index=False)["total_ipc_crimes"]
            .sum()
        )
        
        state_women = (
            women_crime
            .groupby("state_ut", as_index=False)
            .sum(numeric_only=True)
        )
        state_women["total_women_crimes"] = state_women.drop(
            columns=["state_ut"], errors="ignore"
        ).sum(axis=1)
        state_women = state_women[["state_ut", "total_women_crimes"]]
        
        state_children = (
            children_crime
            .groupby("state_ut", as_index=False)["total"]
            .sum()
            .rename(columns={"total": "total_children_crimes"})
        )
        
        # Merge all three
        state_data = (
            state_ipc
            .merge(state_women, on="state_ut", how="inner")
            .merge(state_children, on="state_ut", how="inner")
        )
        
        return state_data
    
    def elbow_analysis(self, X, max_k=8):
        """
        Perform elbow method to find optimal k.
        
        Args:
            X: Scaled feature matrix
            max_k: Maximum number of clusters to test
            
        Returns:
            Dictionary with inertia values for each k
        """
        inertias = []
        silhouettes = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, kmeans.labels_))
        
        return {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouettes": silhouettes
        }
    
    def train(self, n_clusters=4):
        """
        Train KMeans clustering model.
        
        Args:
            n_clusters: Number of clusters (default: 4)
            
        Why 4 clusters?
        - Low crime states
        - Medium-low crime states
        - Medium-high crime states
        - High crime states
        """
        # Load data
        state_data = self.load_data()
        
        # Prepare features
        features = ["total_ipc_crimes", "total_women_crimes", "total_children_crimes"]
        X = state_data[features].values
        
        # Scale features (important for KMeans)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train KMeans
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.model.fit_predict(X_scaled)
        
        # Store results
        self.cluster_centers = self.model.cluster_centers_
        self.silhouette = silhouette_score(X_scaled, self.cluster_labels)
        self.inertia = self.model.inertia_
        
        # Add cluster labels to dataframe
        state_data["cluster"] = self.cluster_labels
        
        # Create cluster descriptions based on centroids
        # Inverse transform centroids to original scale
        centroids_original = self.scaler.inverse_transform(self.cluster_centers)
        
        # Calculate mean crime intensity per cluster
        cluster_intensity = centroids_original.sum(axis=1)
        cluster_ranks = cluster_intensity.argsort().argsort()
        
        cluster_descriptions = {
            0: "Low Crime Intensity",
            1: "Medium-Low Crime Intensity",
            2: "Medium-High Crime Intensity",
            3: "High Crime Intensity"
        }
        
        # Map ranks to descriptions
        state_data["cluster_description"] = state_data["cluster"].map(
            {i: cluster_descriptions[rank] for i, rank in enumerate(cluster_ranks)}
        )
        
        return state_data
    
    def save_results(self, state_data, output_path="data/processed"):
        """
        Save clustering results and model.
        
        Args:
            state_data: DataFrame with cluster assignments
            output_path: Directory to save results
        """
        output_path = Path(output_path)
        
        # Save cluster assignments
        state_data.to_csv(output_path / "clusters.csv", index=False)
        print(f"✓ Saved cluster assignments to {output_path / 'clusters.csv'}")
        
        # Save model
        model_path = Path("models")
        with open(model_path / "kmeans_model.pkl", "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "cluster_centers": self.cluster_centers,
                "silhouette_score": self.silhouette,
                "inertia": self.inertia
            }, f)
        print(f"✓ Saved KMeans model to {model_path / 'kmeans_model.pkl'}")
        
        # Save validation metrics
        metrics = {
            "n_clusters": len(np.unique(self.cluster_labels)),
            "silhouette_score": self.silhouette,
            "inertia": self.inertia,
            "n_states": len(state_data)
        }
        
        with open(output_path / "clustering_metrics.txt", "w") as f:
            f.write("KMeans Clustering Validation Metrics\n")
            f.write("=" * 50 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✓ Saved metrics to {output_path / 'clustering_metrics.txt'}")
        
    def visualize_clusters(self, state_data, save_path="reports"):
        """
        Create visualization of cluster results.
        
        Args:
            state_data: DataFrame with cluster assignments
            save_path: Directory to save plots
        """
        save_path = Path(save_path)
        
        # Cluster distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart: States per cluster
        cluster_counts = state_data["cluster_description"].value_counts()
        axes[0].bar(range(len(cluster_counts)), cluster_counts.values)
        axes[0].set_xticks(range(len(cluster_counts)))
        axes[0].set_xticklabels(cluster_counts.index, rotation=45, ha="right")
        axes[0].set_title("Number of States per Cluster")
        axes[0].set_ylabel("Count")
        
        # Bar chart: Average crimes per cluster
        cluster_summary = state_data.groupby("cluster_description")[
            ["total_ipc_crimes", "total_women_crimes", "total_children_crimes"]
        ].mean()
        
        cluster_summary.plot(kind="bar", ax=axes[1])
        axes[1].set_title("Average Crime Statistics by Cluster")
        axes[1].set_ylabel("Average Count (2001-2014)")
        axes[1].set_xticklabels(cluster_summary.index, rotation=45, ha="right")
        axes[1].legend(["IPC Crimes", "Women Crimes", "Children Crimes"])
        
        plt.tight_layout()
        plt.savefig(save_path / "clustering_visualization.png", dpi=300, bbox_inches="tight")
        print(f"✓ Saved visualization to {save_path / 'clustering_visualization.png'}")
        plt.close()


def main():
    """
    Main execution function for offline clustering.
    Run this ONCE to generate clusters.
    """
    print("\n" + "=" * 60)
    print("Crime Pattern Clustering - OFFLINE TRAINING")
    print("=" * 60 + "\n")
    
    # Initialize model
    clustering_model = CrimeClusteringModel()
    
    # Perform elbow analysis (optional)
    print("Step 1: Loading data...")
    state_data_raw = clustering_model.load_data()
    print(f"  → Loaded {len(state_data_raw)} states/UTs\n")
    
    print("Step 2: Performing elbow analysis...")
    features = ["total_ipc_crimes", "total_women_crimes", "total_children_crimes"]
    X = state_data_raw[features].values
    X_scaled = clustering_model.scaler.fit_transform(X)
    elbow_results = clustering_model.elbow_analysis(X_scaled, max_k=6)
    
    print("  → Elbow Analysis Results:")
    for i, k in enumerate(elbow_results["k_range"]):
        print(f"    k={k}: Inertia={elbow_results['inertias'][i]:.2f}, "
              f"Silhouette={elbow_results['silhouettes'][i]:.3f}")
    print()
    
    # Train with optimal k (4 clusters recommended)
    print("Step 3: Training KMeans with k=4...")
    state_data = clustering_model.train(n_clusters=4)
    print(f"  → Silhouette Score: {clustering_model.silhouette:.3f}")
    print(f"  → Inertia: {clustering_model.inertia:.2f}\n")
    
    # Display cluster summary
    print("Step 4: Cluster Summary:")
    print(state_data.groupby("cluster_description").agg({
        "state_ut": "count",
        "total_ipc_crimes": "mean",
        "total_women_crimes": "mean",
        "total_children_crimes": "mean"
    }).round(2))
    print()
    
    # Save results
    print("Step 5: Saving results...")
    clustering_model.save_results(state_data)
    
    # Visualize
    print("\nStep 6: Creating visualizations...")
    clustering_model.visualize_clusters(state_data)
    
    print("\n" + "=" * 60)
    print("✓ Clustering complete! Results saved to:")
    print("  - data/processed/clusters.csv")
    print("  - models/kmeans_model.pkl")
    print("  - reports/clustering_visualization.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
