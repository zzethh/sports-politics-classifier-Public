# Sports vs. Politics Text Classifier

> **NLU Assignment 1 — Problem 4**  
> **Author:** M25CSA032  
> **Best Model Accuracy:** 96.00% (Bag-of-Words + Naive Bayes)  
> **GitHub Repository:** [sports-politics-classifier-Public](https://github.com/zzethh/sports-politics-classifier-Public)  
> **Live Page:** [Project Page](https://zzethh.github.io/sports-politics-classifier-Public/)

## Overview
This project implements a robust text classification pipeline to distinguish between **Sports** and **Politics** articles using the 20 Newsgroups dataset. It explores various combinations of feature extraction techniques (BoW, TF-IDF, N-grams) and classification algorithms (Naive Bayes, SVM, and Random Forest) to identify the optimal approach.

## Project Structure
- `M25CSA032_prob4.py`: Main script for data loading, training, evaluation, and visualization.
- `M25CSA032_prob4.tex`: Detailed LaTeX report analyzing the findings.
- `nlp_final_results/`: Generated high-resolution plots and results CSV.

## Methodology

### 1. Data Processing
*   **Source**: 20 Newsgroups dataset (`sklearn.datasets.fetch_20newsgroups`).
*   **Categories**:
    *   **Sports**: `rec.sport.hockey`, `rec.sport.baseball`
    *   **Politics**: `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`
*   **Preprocessing**:
    *   Removal of headers, footers, and quoted replies to prevent data leakage.
    *   Lowercase normalization.
    *   Stopword removal (English) to focus on meaningful content.

### 2. Feature Extraction
We experimented with multiple vectorization strategies:
*   **Bag of Words (BoW)**: Simple frequency counts.
*   **TF-IDF**: Weighted importance to penalize common terms.
*   **N-grams**: Unigrams, Bigrams, and Trigrams to capture phrase-level context.

### 3. Classification Models
*   **Naive Bayes (Multinomial)**: Simple probabilistic baseline that performed best.
*   **Support Vector Machine (SVM)**: Linear kernel for high-dimensional separation.
*   **Random Forest**: Ensemble method to capture non-linear patterns.

## Results & Analysis

We trained and evaluated **9 different configurations**. The results highlight the effectiveness of simple, robust baselines for distinct topic classification.

| Feature Set | Classifier | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Bag of Words** | **Naive Bayes** | **96.00%** | **0.96** |
| N-gram (1,2) | Naive Bayes | 95.89% | 0.96 |
| TF-IDF | Naive Bayes | 95.56% | 0.96 |
| TF-IDF | SVM | 95.35% | 0.95 |
| N-gram (1,2) | Random Forest | 94.16% | 0.94 |
| TF-IDF | Random Forest | 93.29% | 0.93 |
| Bag of Words | Random Forest | 93.07% | 0.93 |
| N-gram (1,2) | SVM | 91.77% | 0.92 |
| Bag of Words | SVM | 91.67% | 0.92 |

### Key Findings
1.  **Simplicity Wins**: The simplest model (`BoW + Naive Bayes`) achieved the highest accuracy (**96.00%**).
2.  **Stopwords Matter**: Removing stopwords significantly reduced noise, allowing the models to focus on topic-specific keywords like "game", "team" (Sports) vs "government", "rights" (Politics).
3.  **TF-IDF vs BoW**: While TF-IDF is usually superior, BoW edged it out here, likely because the two topics have very distinct vocabularies where raw frequency is a strong enough signal.

## Visualizations
The script generates 7 high-quality visualizations in `nlp_final_results/`, including:
*   **Confusion Matrix**: To pinpoint misclassifications.
*   **Model Ranking**: A horizontal bar chart comparing all 15 configurations.
*   **Top Keywords**: Identifying the most predictive words for each class.
*   **Class Balance**: Donut chart verifying dataset distribution.

## How to Run

### Prerequisites
```bash
pip install scikit-learn pandas matplotlib seaborn numpy
```
*(On the cluster, use `module load python/3.10`)*

### Execution
Run the script to train models and generate plots:
```bash
python3 M25CSA032_prob4.py
```
*The script ends with an interactive mode where you can type sentences to test the classifier in real-time.*
