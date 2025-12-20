# Quantitative Analysis for Agricultural Strategy Optimization

> **2024 Contemporary Undergraduate Mathematical Contest in Modeling (CUMCM) - Problem C**
>
> 🏆 **National Second Prize** (Ranked Top **2.53%** among undergraduate teams nationwide)

### About CUMCM
The Contemporary Undergraduate Mathematical Contest in Modeling (CUMCM, also known as China Undergraduate Mathematical Contest in Modeling) was founded in 1992 and is held annually. It is one of the first 19 competitions included in the national "University Discipline Competition Ranking." In 2025, the contest attracted over **200,000 participants** from **68,311 teams** (61,463 undergraduate and 6,848 vocational teams) representing **1,837 institutions** across China, the US, UK, Canada, Finland, Malaysia, and other countries.
🔗 [Official Website](https://www.mcm.edu.cn/)

---

## 🏞 Background

This project addresses a real-world agricultural planning problem for a village in the mountainous area of North China. The goal is to optimize planting strategies from **2024 to 2030** to maximize economic returns while ensuring ecological sustainability.

The agricultural setting is highly heterogeneous:
* **Land Resources:** 1,201 mu of arable land dispersed into **34 plots** of varying types (Flat Dry Land, Terraced Fields, Hill Slopes, Irrigated Land) and **20 Greenhouses** (16 Ordinary, 4 Smart).
* **Agronomic Rules:**
    * **Crop Rotation:** Continuous cropping of the same crop on the same plot is prohibited to prevent yield loss.
    * **Soil Improvement:** Leguminous crops must be planted on *every* plot at least once within any **3-year rolling window**.
    * **Operational Constraints:** Planting areas must not be too dispersed or too small to facilitate management.

---

## 🚀 The Challenge

The core difficulty lies in solving a **Large-Scale Multi-Stage Stochastic Optimization** problem under complex constraints. This is not a toy problem; it involves high-dimensional combinatorics and market uncertainty.

### 1. High Dimensionality & Complexity
* **Decision Variables:** We optimized the allocation of **41 different crops** across **54 specific plots** over **7 years** (with 1-2 seasons per year). This results in a Mixed-Integer Linear Programming (MILP) model with over **50,000+ decision variables** ($x_{i,j,s,t}$) and binary indicators.
* **Complex Constraints:**
    * **Temporal Dependencies:** The "Legume Rotation" constraint creates a dependency chain linking decision variables across 3-year sliding windows ($t, t+1, t+2$).
    * **Heterogeneous Logic:** Different constraints apply to different land types (e.g., Smart Greenhouses vs. Hill Slopes have different seasonal logic).

### 2. Non-Linearity & Discontinuities
* **Piecewise Revenue Functions:** In Problem 1, excess production is sold at a discounted rate (or wasted), creating a non-linear objective function. We linearized this using **Big-M methods** and auxiliary binary variables.
* **Threshold Constraints:** Minimum area requirements introduce semi-continuous constraints.

### 3. Uncertainty & Market Coupling 
* **Stochastic Factors:** Yields, costs, and prices follow different stochastic processes (Growth Trends vs. Mean-Reverting Random Walks).
* **Market Correlation:** In Problem 3, crops are not independent. Price changes in *Wheat* affect the demand for *Corn* (Substitute) and *Vegetables* (Complement). We had to model this **"Butterfly Effect"** without historical correlation data by deriving a **Cross-Price Elasticity Matrix**.

---
## 🛠 Our Solution Framework

We constructed a **Three-Stage Quantitative System** that integrates statistical learning, stochastic simulation, and large-scale operations research.

### 1. High-Dimensional Feature Engineering & Clustering
> **Goal:** Reduce the dimensionality of 41 crops into tractable strategic groups while preserving economic characteristics.
* **Feature Selection:** Conducted **Pearson Correlation Analysis** to identify and remove highly collinear features, selecting independent indicators (e.g., *Yield Stability*, *Cost-Profit Ratio*) that maximize information gain.
* **Unsupervised Learning:** Adopted **Hierarchical Clustering (Agglomerative)** instead of simple K-Means to better capture the nested structure of agricultural data.
    * **Model Selection:** Utilized the **Silhouette Score** and **Calinski-Harabasz Index** to mathematically determine the optimal number of clusters ($k=4$), avoiding subjective bias.
    * **Visualization:** Applied **Principal Component Analysis (PCA)** to project high-dimensional clusters into 3D space, verifying the separability and compactness of the resulting crop categories (e.g., "High-Risk High-Reward" vs. "Stable Grain Security").

### 2. Deterministic Optimization (Large-Scale MILP)
> **Goal:** Solve a combinatorial allocation problem with 50,000+ variables under complex agronomic constraints.
* **Model Formulation:** Built a **Mixed-Integer Linear Programming (MILP)** model.
* **Constraint Handling:**
    * **Linearization:** Used the **Big-M Method** and auxiliary binary variables to linearize the non-continuous "Discounted Oversupply" revenue function in Problem 1-2.
    * **Multi-Objective Optimization:** Implemented a **Two-Stage Solving Strategy**. Stage 1 maximizes profit, while Stage 2 minimizes the **Fragmentation Index** (number of crop-plot switches) to ensure operational feasibility, constrained to retain $\ge 95\%$ of Stage 1 optimality.

### 3. Stochastic Modeling & Climate Regression
> **Goal:** Quantify uncertainty in yields and prices over a 7-year horizon.
* **Climate Impact Analysis:** Built a **Multiple Linear Regression** model, identifying *Winter Sunshine Duration* as the statistically significant predictor ($p < 0.05, R^2 > 0.82$) for Wheat and Corn yields.
* **Hybrid Monte Carlo Simulation:**
    * Modeled yield volatility using **Mean-Reverting Random Walks**.
    * Simulated price evolution using **Compound Growth Models** with randomized drift and volatility parameters derived from historical variance.
    * Generated **1,000+ scenarios** to construct confidence intervals for future agricultural output.

### 4. Coupled Market Dynamics (Structural Estimation)
> **Goal:** Model the "Butterfly Effect" where a price surge in one crop shifts demand for others.
* **Elasticity Matrix Construction:** Instead of purely data-driven regression (which lacks sufficient historical data), we employed a **Feature-Based Economic Derivation** approach.
    * Quantified **Substitution Effects** (positive elasticity) for crops within the same dietary class.
    * Quantified **Complementarity Effects** (negative elasticity) for cross-category combinations.
* **Price-Demand Coupling:** Integrated **Geometric Brownian Motion (GBM)** for continuous price simulation, linking it to demand via the derived $41 \times 41$ Cross-Price Elasticity Matrix ($\Delta \ln Q_{i} = \sum E_{ij} \cdot \Delta \ln P_{j}$).

### 5. Robustness & Sensitivity Analysis
* **Perturbation Testing:** Injected **Gaussian White Noise** ($N(0, \sigma^2)$) into input parameters (Price/Yield) with varying intensities ($\sigma \in [0.01, 0.10]$).
* **Stability Metrics:** Calculated the **Profit Sensitivity Coefficient** and **Mean Squared Error (MSE)** of the decision variables to prove the strategy's resilience against market volatility.

---
## 🧰 Tech Stack

### Operations Research & Optimization
* **Solvers:** `Gurobi Optimization` (Large-scale MILP solving, MIPGap tuning)
* **Methods:** Mixed-Integer Linear Programming, Big-M Constraint Relaxation, Multi-Objective Optimization (Lexicographic method), Sensitivity Analysis.

### Statistical Learning & Data Mining
* **Libraries:** `SciPy` (Hierarchical Clustering, Dendrograms), `Scikit-learn` (PCA, StandardScaler, Silhouette Analysis).
* **Algorithms:** Agglomerative Clustering, Principal Component Analysis (PCA), Multiple Linear Regression (OLS), Pearson/Spearman Correlation.

### Stochastic Modeling & Simulation
* **Techniques:** Monte Carlo Simulation (1000+ paths), Geometric Brownian Motion (GBM), Gaussian Noise Injection, Feature-based Matrix Derivation.
* **Metrics:** R-squared ($R^2$), Mean Squared Error (MSE), Confidence Intervals.

### Data Engineering & Visualization
* **Libraries:** `Pandas` (Advanced indexing, Pivot tables), `Seaborn` & `Matplotlib` (Heatmaps, 3D Scatter plots, Regression plots).

---
## 🎓 Acknowledgements

First and foremost, we extend our deepest gratitude to our supervisor, **Prof. Ling Xue**, whose invaluable guidance and insightful feedback were the compass guiding us through this complex challenge. Her mentorship not only helped us navigate the technical difficulties but also inspired us to think deeply about the practical implications of our models.

Equally, this achievement belongs to the unwavering spirit of our team. During those four intense days and nights, **Sihan Lyu**, **Xinyahui Zhao** and I fought side by side, turning abstract mathematical problems into concrete code and prose. The shared memories of debugging at dawn, debating modeling strategies, and supporting each other under extreme pressure will be cherished forever as a highlight of our undergraduate journey.

---
## 📂 Repository Structure

```bash
agricultural-planting-optimization/
│
├── 📁 notebooks/               # 12 Jupyter notebooks containing full source code
│   ├── 01_data_preprocessing/  # Clustering, EDA, and Data Cleaning
│   ├── 02_problem1_solution/   # MILP for Deterministic Scenarios (Base & Discounted)
│   ├── 03_problem2_solution/   # Stochastic Optimization & Climate Regression
│   ├── 04_problem3_solution/   # Coupled Market Dynamics (Elasticity Matrix)
│   └── 05_robustness/          # Sensitivity Analysis (Noise Injection)
│
├── 📁 data/
│   ├── raw/                    # Original problem datasets
│   ├── processed/              # Clustered data & Monte Carlo predictions
│   └── results/                # Final Strategy Outputs (Excel)
│
├── 📁 paper/                   # Research Paper & Figures
│   ├── CN/                     # Chinese Version
│   ├── EN/                     # English Version
│
├── 📁 award/                  # Certificates
├── 📁 verification/            # Guides to verify the award and the ranking
└── README.md
