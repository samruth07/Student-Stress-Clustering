# Student Stress Pattern Analysis Using Unsupervised Learning

A Python-based unsupervised machine learning project that explores patterns in student stress-related survey data. The project applies three clustering algorithms to 20 student-related features and compares the resulting groups using internal evaluation metrics, dimensionality-reduction visualizations, cluster agreement analysis, and post-clustering validation.

The `stress_level` column is held out from model fitting. It is used only after clustering to examine how the discovered groups relate to the dataset's existing stress-level categories.

## 🚀 Project Overview

This project analyzes **1,100 student records** containing **20 predictor features** and one held-out target column. The features describe psychological, physiological, environmental, academic, and social factors.

The following clustering algorithms are implemented and compared:

- K-Means Clustering
- Agglomerative Hierarchical Clustering
- DBSCAN

Dimensionality reduction and visualization are performed using:

- Principal Component Analysis (PCA)
- t-distributed Stochastic Neighbor Embedding (t-SNE)
- Stochastic Neighbor Embedding (SNE)

## ✨ Key Features

- **Exploratory Data Analysis:** Examines feature distributions, missing values, descriptive statistics, and feature correlations.
- **Data Preprocessing:** Separates the held-out label, handles missing values, and standardizes predictor features.
- **Clustering Algorithms:** Compares K-Means, Agglomerative Hierarchical Clustering, and DBSCAN using the standardized feature matrix.
- **Cluster Selection:** Uses the Elbow Method and Silhouette Score to examine candidate K-Means cluster counts.
- **DBSCAN Parameter Tuning:** Evaluates combinations of `eps` and `min_samples`, considering both cluster quality and the proportion of records assigned to clusters.
- **Dimensionality Reduction:** Uses PCA, t-SNE, and SNE to explore the data in two dimensions.
- **Model Evaluation:** Reports Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
- **Cluster Agreement:** Uses pairwise Adjusted Rand Index (ARI) to compare the partitions produced by the three algorithms.
- **Post-clustering Validation:** Compares cluster labels with the held-out `stress_level` column using ARI and Normalized Mutual Information (NMI).
- **Result Export:** Saves figures, evaluation tables, cluster assignments, embeddings, and experiment metadata.

## 📂 Dataset Description

**Dataset:** Student Stress Factors Dataset

| Property | Details |
|---|---:|
| Records | 1,100 |
| Total columns | 21 |
| Predictor features | 20 |
| Held-out label | `stress_level` |
| Storage format | CSV |
| Stored data type | Integer-coded values (`int64`) |

### Features Used for Clustering

The following 20 features are used as model inputs:

**Psychological factors**
- `anxiety_level`
- `self_esteem`
- `mental_health_history`
- `depression`

**Physiological factors**
- `headache`
- `blood_pressure`
- `sleep_quality`
- `breathing_problem`

**Environmental factors**
- `noise_level`
- `living_conditions`
- `safety`
- `basic_needs`

**Academic factors**
- `academic_performance`
- `study_load`
- `teacher_student_relationship`
- `future_career_concerns`

**Social factors**
- `social_support`
- `peer_pressure`
- `extracurricular_activities`
- `bullying`

The `stress_level` column is excluded from preprocessing, dimensionality reduction, and clustering. It is used only for post-clustering validation. The columns are integer-coded scores or ratings; their numeric ordering is retained for the baseline distance-based experiments.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Scientific computing | SciPy |
| Data visualization | Matplotlib, Seaborn |
| Development environment | Jupyter Notebook, Visual Studio Code |

## ⚙️ Project Workflow

1. **Data Loading:** Load the CSV file and inspect its dimensions, columns, data types, and values.
2. **Exploratory Data Analysis:** Examine feature distributions, descriptive statistics, correlations, missing values, and possible unusual values.
3. **Feature Preparation:** Separate the 20 predictor features from the held-out `stress_level` column.
4. **Preprocessing:** Apply median imputation where required and standardize the predictor features using `StandardScaler`.
5. **K-Means Cluster Selection:** Examine inertia and Silhouette Scores for candidate values of K.
6. **Model Training:** Fit K-Means and Agglomerative Hierarchical Clustering.
7. **DBSCAN Tuning:** Examine nearest-neighbour distances and evaluate candidate `eps` and `min_samples` settings.
8. **Model Evaluation:** Compare internal clustering metrics and report DBSCAN's coverage and noise percentage.
9. **Dimensionality Reduction:** Generate PCA, t-SNE, and SNE embeddings and visualize cluster assignments.
10. **Cluster Analysis:** Examine cluster sizes, feature profiles, and agreement between algorithms.
11. **Post-clustering Validation:** Compare the discovered clusters with `stress_level` without using that column during model fitting.
12. **Result Saving:** Export figures, tables, embeddings, cluster assignments, and metadata.

## 🧠 Algorithms Implemented

### 1. K-Means Clustering

K-Means partitions observations into a specified number of clusters by minimizing within-cluster squared distances.

- Candidate cluster counts: K = 2 to 10
- Initialization: `k-means++` (scikit-learn default)
- Random state: 42
- Initializations: 20 for the main model
- Selected cluster count in the recorded run: **K = 5**

### 2. Agglomerative Hierarchical Clustering

Agglomerative clustering builds groups by progressively merging observations or clusters. Ward linkage is used with Euclidean distance.

- Number of clusters: 5
- Linkage method: Ward

### 3. DBSCAN

DBSCAN identifies dense regions and marks observations that do not meet the density requirements as noise (`-1`).

The parameter search considers several `min_samples` values and candidate `eps` values derived from nearest-neighbour distance quantiles. The selected configuration in the recorded run is:

- `eps = 1.809`
- `min_samples = 15`
- Clusters excluding noise: 4
- Records assigned to clusters: 81%
- Records classified as noise: 19%

## 📊 Model Evaluation

The clustering algorithms are evaluated using three internal metrics:

- **Silhouette Score:** Higher values generally indicate better separation and cohesion.
- **Davies-Bouldin Index:** Lower values generally indicate better separation between clusters.
- **Calinski-Harabasz Score:** Higher values generally indicate greater between-cluster dispersion relative to within-cluster dispersion.

### Full-Dataset Algorithm Comparison

The following values are from the recorded experiment. For DBSCAN, internal metrics are calculated after excluding noise observations; its coverage and noise percentage are shown separately. Therefore, DBSCAN's metric values are not directly comparable with full-coverage model scores without considering this difference.

| Algorithm | Clusters | Coverage (%) | Noise (%) | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|---:|---:|---:|
| K-Means | 5 | 100.0 | 0.0 | 0.4489 | 2.3097 | 560.2014 |
| Agglomerative | 5 | 100.0 | 0.0 | 0.3667 | 1.7272 | 549.2617 |
| DBSCAN | 4 | 81.0 | 19.0 | 0.4832 | 1.1294 | 1796.9997 |

### Comparison on the DBSCAN Non-noise Subset

To compare the algorithms on the same observations, the metrics were recalculated using the 891 records assigned to DBSCAN clusters.

| Algorithm | Clusters on subset | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|---:|
| K-Means | 3 | 0.5706 | 0.6878 | 2332.5922 |
| Agglomerative | 4 | 0.4832 | 1.1294 | 1796.9997 |
| DBSCAN | 4 | 0.4832 | 1.1294 | 1796.9997 |

### Post-clustering Validation

The `stress_level` column is compared with the generated cluster labels only after model fitting. It is not used as an input to clustering or dimensionality reduction.

| Algorithm | Adjusted Rand Index (ARI) | Normalized Mutual Information (NMI) |
|---|---:|---:|
| K-Means | 0.7166 | 0.6792 |
| Agglomerative | 0.6096 | 0.6763 |
| DBSCAN | 0.5990 | 0.6692 |

K-Means has the highest ARI and NMI among the three algorithms in this recorded post-clustering comparison. These scores measure agreement with the supplied label; they do not establish clinical or psychological validity.

### Pairwise Cluster Agreement

Pairwise Adjusted Rand Index comparisons show that:

- K-Means and Agglomerative Clustering have an ARI of 0.8337 on the full dataset and 0.8712 on the DBSCAN non-noise subset.
- K-Means and DBSCAN have an ARI of 0.8165 on the full dataset and 0.8712 on the DBSCAN non-noise subset.
- Agglomerative Clustering and DBSCAN have an ARI of 0.9819 on the full dataset and 1.0000 on the DBSCAN non-noise subset.

### K-Means Stability

Across repeated runs with different random seeds, the mean pairwise Adjusted Rand Index for K-Means was **0.9383**, indicating consistent assignments across the tested runs.

## 📉 Dimensionality Reduction

### Principal Component Analysis (PCA)

PCA is applied to the standardized 20-feature matrix.

- Components required to retain at least 95% of the variance: **16**
- Variance explained by the first two principal components: **65.54%**

The notebook generates a scree plot, cumulative explained-variance plot, two-dimensional cluster visualizations, and a loading heatmap.

### t-SNE

t-SNE is used to visualize local neighbourhood structure in two dimensions. Its embedding is used for visualization, not as the primary feature space for evaluating clustering quality.

### SNE

A separate SNE implementation uses Gaussian conditional neighbour probabilities and a two-dimensional embedding optimized by minimizing a KL-divergence objective. To keep computation manageable, SNE uses a reproducible sample of **800 records**.

Recorded SNE run:

- Converged: True
- Iterations: 106
- Function evaluations: 111
- Final KL objective: 1168.6407
- Final gradient norm: 0.0013961
- All embedding coordinates finite: True

SNE and t-SNE plots are used to inspect neighbourhood patterns; apparent separation in a two-dimensional embedding should not be interpreted as proof of globally separated clusters.

## 📂 Project Structure

```text
Student-Stress-Clustering/
│
├── StressLevelDataset.csv
├── Student_Stress_Clustering.ipynb
├── README.md
├── Unsupervised Clustering of Student Stress Patterns Report.docx
│
└── outputs_student_stress/
    ├── figures/
    │   ├── feature_distributions.png
    │   ├── spearman_correlation_heatmap.png
    │   ├── feature_boxplots.png
    │   ├── target_distribution_posthoc_only.png
    │   ├── kmeans_elbow_and_silhouette.png
    │   ├── dbscan_k_distance.png
    │   ├── silhouette_comparison.png
    │   ├── davies_bouldin_comparison.png
    │   ├── calinski_harabasz_comparison.png
    │   ├── cluster_size_comparison.png
    │   ├── kmeans_cluster_profile_heatmap.png
    │   ├── pca_explained_variance.png
    │   ├── pca_cluster_comparison.png
    │   ├── pca_loadings_heatmap.png
    │   ├── tsne_cluster_comparison.png
    │   ├── original_sne_cluster_comparison.png
    │   ├── pairwise_ari_comparison.png
    │   └── dbscan_noise_distribution.png
    │
    ├── tables/
    │   ├── feature_summary.csv
    │   ├── preprocessing_summary.csv
    │   ├── kmeans_k_selection.csv
    │   ├── dbscan_tuning_results.csv
    │   ├── clustering_model_comparison.csv
    │   ├── cluster_sizes.csv
    │   ├── pca_loadings_pc1_pc2.csv
    │   ├── original_vs_pca95_kmeans.csv
    │   ├── tsne_embedding.csv
    │   ├── original_sne_embedding.csv
    │   ├── original_sne_diagnostics.csv
    │   ├── kmeans_cluster_profile_means.csv
    │   ├── kmeans_cluster_profile_medians.csv
    │   ├── posthoc_external_validation.csv
    │   ├── pairwise_ari_comparison.csv
    │   ├── dbscan_non_noise_subset_comparison.csv
    │   └── dbscan_noise_distribution.csv
    │
    ├── final_cluster_assignments.csv
    ├── final_algorithm_comparison.csv
    ├── final_cluster_sizes.csv
    ├── final_pairwise_ari_comparison.csv
    ├── final_dbscan_non_noise_comparison.csv
    └── model_metadata.json
```

The output directory and its contents are generated when the notebook is executed. Some conditional files, such as the missing-value plot, are created only when applicable. Additional CSV files may be generated by individual analysis cells.

## 💻 Installation and Execution

### 1. Install Python

Use Python 3.10 or later.

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter ipykernel
```

### 3. Place the dataset

Keep `StressLevelDataset.csv` in the same directory as the notebook, or place it in a `data/` or `dataset/` subdirectory.

### 4. Open the notebook

Open `Student_Stress_Clustering.ipynb` in Jupyter Notebook or Visual Studio Code and select the Python environment containing the required libraries.

### 5. Run the notebook

Execute the cells from top to bottom. The notebook creates the `outputs_student_stress/` directory and saves the generated results there.

## 📌 Applications

- Exploring patterns across student stress-related survey factors.
- Comparing groups identified by different unsupervised clustering algorithms.
- Examining relationships among psychological, physiological, environmental, academic, and social variables.
- Visualizing high-dimensional survey data using dimensionality-reduction techniques.
- Demonstrating clustering evaluation, parameter tuning, and post-clustering validation.

The analysis is exploratory and is not intended for diagnosis or individual assessment of mental health.

## 🔮 Future Enhancements

- Validate feature coding and ordinal scales against the full dataset documentation.
- Examine alternative distance measures suitable for ordinal and mixed-type data.
- Test clustering stability across a wider range of parameter settings and random seeds.
- Compare additional DBSCAN settings and density-based alternatives.
- Examine t-SNE and SNE sensitivity to their hyperparameters.
- Extend the analysis with additional student datasets to assess reproducibility.

## ⚠️ Limitations

- The dataset contains 1,100 records and 20 predictor features; findings may not generalize to other student populations.
- Predictors are integer-coded scores or ratings. Their ordering is retained for the baseline distance-based experiments, so results depend on the appropriateness of that representation.
- The original survey collection date is not established by the CSV alone.
- Cluster identifiers are arbitrary and do not inherently represent low, medium, or high stress.
- Internal clustering metrics describe geometric properties of the fitted partitions and do not establish real-world or clinical validity.
- DBSCAN's internal metrics exclude noise observations; coverage and noise percentage must be considered when interpreting its scores.
- SNE is run on a reproducible subset of 800 records due to computational cost.
- PCA, t-SNE, and SNE visualizations are projections and may not preserve every relationship in the original feature space.

## 👨‍💻 Project Information

| Property | Details |
|---|---|
| Project title | Unsupervised Clustering of Student Stress Patterns |
| Domain | Machine Learning / Unsupervised Learning |
| Dataset size | 1,100 records, 21 columns |
| Clustering input | 20 predictor features |
| Clustering algorithms | K-Means, Agglomerative Hierarchical Clustering, DBSCAN |
| Dimensionality reduction | PCA, t-SNE, SNE |
| Evaluation | Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI |
| Development environment | Python, Jupyter Notebook, Visual Studio Code |
```
