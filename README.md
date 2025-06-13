# Exploring Data Clustering Techniques: A Comparative Study Using California Housing Data

![GitHub top language](https://img.shields.io/github/languages/top/zeynepcindemir/California-Housing-Data-Mining-Project?style=flat-square)

This repository contains the code and resources for a comprehensive comparative study of various data clustering algorithms applied to the California Housing 1990 census dataset.

## Table of Contents
- [Abstract](#abstract)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Clustering Algorithms](#clustering-algorithms)
  - [Evaluation Metrics](#evaluation-metrics)
- [Results and Discussion](#results-and-discussion)
  - [Performance Comparison](#performance-comparison)
  - [Cluster Visualizations](#cluster-visualizations)
  - [Key Findings](#key-findings)
- [How to Run the Project](#how-to-run-the-project)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Conclusion](#conclusion)

## Abstract
This study examines the performance of various clustering algorithms on the California Housing 1990 census dataset by analyzing metrics such as the Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, time consumed, and memory usage. In addition to performance evaluation, this project visualizes the clustering results and conducts a detailed Exploratory Data Analysis (EDA) on the dataset. The primary goal is to identify the most effective clustering techniques in terms of both accuracy and efficiency, providing insights into the suitability of each algorithm for real-world data applications. This comparative analysis aims to guide data scientists in choosing appropriate clustering methods for large-scale housing data.

## Dataset
- **Source**: The dataset used is the **California Housing 1990 census dataset**, sourced from a public GitHub repository maintained by Aurélien Geron.
- **Description**: This dataset includes detailed information on 20,640 housing units across California from the 1990 census.
- **Attributes**: The 10 attributes include geographical coordinates (longitude, latitude) and housing-specific details such as:
  - `housing_median_age`
  - `total_rooms`
  - `total_bedrooms`
  - `population`
  - `households`
  - `median_income`
  - `median_house_value`
  - `ocean_proximity` (Categorical)

## Methodology

### Exploratory Data Analysis (EDA)
A comprehensive EDA was conducted to prepare the dataset for clustering. Key steps included:
- **Initial Inspection**: Assessing dimensions, data types, and descriptive statistics.
- **Visualization**: Using histograms and bar charts to understand feature distributions.
- **Handling Missing Values**: Imputing missing `total_bedrooms` values using the median.
- **Data Encoding**: Applying `OneHotEncoder` to the `ocean_proximity` attribute.
- **Feature Scaling**: Standardizing all numerical features using `StandardScaler`.
- **Correlation Analysis**: Creating a correlation matrix to understand feature interdependencies.

<p align="center">
  <img src="https://raw.githubusercontent.com/zeynepcindemir/California-Housing-Data-Mining-Project/main/corr.png" width="45%">
  <img src="https://raw.githubusercontent.com/zeynepcindemir/California-Housing-Data-Mining-Project/main/density.png" width="45%">
</p>
<p align="center"><i>Correlation Matrix and Geographical Density Plot</i></p>

### Clustering Algorithms
Eleven different clustering algorithms were implemented and evaluated:
1.  **K-Means**
2.  **K-Medoids**
3.  **K-Medians**
4.  **Mini Batch K-Means**
5.  **Mean Shift**
6.  **Gaussian Mixture Model (GMM)**
7.  **Agglomerative Clustering**
8.  **BIRCH**
9.  **OPTICS**
10. **HDBSCAN**
11. **DBSCAN**

For each algorithm, extensive hyperparameter tuning was performed using a grid search methodology to find the optimal settings for this specific dataset.

### Evaluation Metrics
The performance of each algorithm was assessed using intrinsic metrics that do not require ground truth labels:
- **Silhouette Score**: Measures how similar a data point is to its own cluster compared to other clusters. *Higher is better*.
- **Davies-Bouldin Index**: Measures the ratio of within-cluster scatter to between-cluster separation. *Lower is better*.
- **Calinski-Harabasz Index**: Measures the ratio of between-cluster variance to within-cluster variance. *Higher is better*.
- **Time and Memory Usage**: To evaluate computational efficiency.

## Results and Discussion

### Performance Comparison
The algorithms were evaluated after reducing the data's dimensionality using PCA (with 2, 3, and 4 components). The tables below summarize the performance metrics for each algorithm under its optimal hyperparameter settings.

<p align="center">
  <img src="https://raw.githubusercontent.com/zeynepcindemir/California-Housing-Data-Mining-Project/main/df1.png" alt="Performance metrics for K-Means, K-Medians, K-Medoids, and Mini Batch K-Means" width="80%">
  <br>
  <i>Table 1: Performance of K-Means Variants</i>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/zeynepcindemir/California-Housing-Data-Mining-Project/main/df2.png" alt="Performance metrics for other clustering algorithms" width="80%">
  <br>
  <i>Table 2: Performance of Other Clustering Algorithms</i>
</p>

### Cluster Visualizations
Visualizing the clusters provides an intuitive understanding of each algorithm's behavior.

<p align="center">
  <img src="https://raw.githubusercontent.com/zeynepcindemir/California-Housing-Data-Mining-Project/main/agg.png" width="45%">
  <img src="https://raw.githubusercontent.com/zeynepcindemir/California-Housing-Data-Mining-Project/main/hdbscan.png" width="45%">
  <br>
  <i>Agglomerative Clustering (left) and HDBSCAN (right)</i>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/zeynepcindemir/California-Housing-Data-Mining-Project/main/gausp.png" width="45%">
  <img src="https://raw.githubusercontent.com/zeynepcindemir/California-Housing-Data-Mining-Project/main/dbscan.png" width="45%">
  <br>
  <i>Gaussian Mixture Model (left) and DBSCAN (right)</i>
</p>

### Key Findings
- **Best Overall Quality**: **Agglomerative Clustering** consistently achieved the highest Silhouette Scores and lowest Davies-Bouldin scores, indicating it formed the most well-separated and compact clusters.
- **Best Cluster Definition**: **K-Means variants** scored highest on the Calinski-Harabasz Index, suggesting they are excellent at maximizing cluster separation.
- **Resource Efficiency**: **DBSCAN**, **HDBSCAN**, and **BIRCH** were the most memory-efficient algorithms, making them suitable for resource-constrained environments.
- **Time Consumption**: **K-Means** was the fastest, while **Mean Shift** and **K-Medoids** were the most time-consuming due to their computational complexity.

## How to Run the Project

### Prerequisites
You need Python 3.x and the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `hdbscan`

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/zeynepcindemir/California-Housing-Data-Mining-Project.git
    cd California-Housing-Data-Mining-Project
    ```
2.  Install the required packages. It is recommended to use a virtual environment.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn hdbscan jupyter
    ```

### Usage
All the analysis is contained within a single Jupyter Notebook. To run the project:
1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open the `Data_Mining_Final_Project.ipynb` file and run the cells sequentially.

## Conclusion
This study provides a detailed, practical comparison of eleven clustering algorithms on a real-world dataset. The findings highlight a clear trade-off between clustering quality and computational resources. **Agglomerative Clustering** is recommended for applications demanding high-quality clusters, while **K-Means** and its variants offer a strong balance of performance and speed. Density-based methods like **HDBSCAN** and **DBSCAN** excel in handling noise and are memory-efficient. This analysis serves as a practical guide for data scientists to select the appropriate clustering algorithm based on their specific goals and constraints.

Future work could involve exploring more recent clustering algorithms, applying these methods to different types of datasets, and investigating the impact of different feature engineering techniques.
