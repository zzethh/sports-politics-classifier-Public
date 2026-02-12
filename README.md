## NLU Assignment 1 – Zenith (M25CSA032)

This repository contains my solutions for **CSL 7640: Natural Language Understanding – Assignment 1**.  
Each problem is implemented in a separate Python file following the required naming conventions.

---

### Problem 1 – Reggy++ (`M25CSA032_prob1.py`, `M25CSA032_prob1.log`, `M25CSA032_prob1.txt`)

- **Goal**: Extend the in‑class *Reggy* chatbot using only regular expressions and basic string processing.
- **Features implemented**:
  - Asks for the user’s birthday in multiple formats (numeric and text month) and computes age.
  - Asks for mood and responds appropriately, handling small typos and repeated characters.
  - Extracts the surname from a full name (last word heuristic).
  - Interactive loop so the chatbot can be run multiple times to observe its “naturalness”.
- **How to run**:
  - `python3 M25CSA032_prob1.py`
- **Deliverables**:
  - `M25CSA032_prob1.py` – chatbot implementation.
  - `M25CSA032_prob1.log` – transcripts from multiple runs, including successful, typo‑heavy and failure cases.
  - `M25CSA032_prob1.txt` – reflection (300–500 words) on naturalness, strengths and limitations of regex‑based interaction.

---

### Problem 2 – Byte Pair Encoding (`M25CSA032_prob2.py`, `corpus.txt`)

- **Goal**: Implement **Byte Pair Encoding (BPE) tokenization from scratch** using only standard Python libraries.
- **Main steps**:
  - Read a training corpus from `corpus.txt` (one sentence / word sequence per line).
  - Build a character‑level vocabulary with explicit end‑of‑word markers `</w>`.
  - Repeatedly merge the most frequent adjacent symbol pair **K** times.
  - Output the final set of learned subword tokens.
- **How to run**:
  - `python3 M25CSA032_prob2.py K corpus.txt`  
    where **K** is the number of merge operations.
- **Corpus**:
  - `corpus.txt` contains several groups of related words (comparatives, verb forms, derivations) so that the learned merges are interpretable.

---

### Problem 3 – Naive Bayes Sentiment Classifier (`M25CSA032_prob3.py`, `pos.txt`, `neg.txt`)

- **Goal**: Build a **Naive Bayes sentiment classifier from scratch**.
- **Pipeline**:
  - Read positive sentences from `pos.txt` and negative sentences from `neg.txt` (one per line).
  - Tokenize with simple whitespace splitting and lowercasing (as required).
  - Estimate priors and word likelihoods with **Laplace (+1) smoothing**.
  - Use an 80/20 train–validation split to check performance.
  - Enter an interactive mode where the user can type a sentence and receive a `POSITIVE` or `NEGATIVE` prediction.
- **How to run**:
  - `python3 M25CSA032_prob3.py`
  - After training, follow the on‑screen prompt to input sentences or type `exit` to quit.

---

### Problem 4 – Sports vs Politics Classification (`M25CSA032_prob4.py`, `M25CSA032_prob4.tex`, `nlp_final_results/`)

- **Goal**: Design a **binary topic classifier** that distinguishes between Sports and Politics documents using modern ML techniques.
- **Data & features**:
  - Uses a subset of the 20 Newsgroups dataset (sports and politics groups only).
  - Compares several feature representations: Bag of Words, Bag of Words bigrams, TF‑IDF, TF‑IDF bigrams/trigrams, and Count‑based N‑grams (1,2).
  - Evaluates multiple models: Multinomial Naive Bayes, Linear SVM, Logistic Regression, Random Forest, and Gradient Boosting.
- **Outputs**:
  - A CSV file `nlp_final_results/final_results.csv` with accuracy and weighted F1 for each feature–model configuration.
  - Visualisations saved to `nlp_final_results/`, including:
    - Top‑word bar charts for Sports and Politics.
    - A donut chart of class distribution.
    - A horizontal bar chart ranking all configurations by accuracy.
    - A confusion matrix and radar plot for the best‑performing model.
  - A detailed LaTeX report:
    - `M25CSA032_prob4.tex` – describes data collection, feature design, models, results, quantitative comparisons, and limitations.
    - Can be compiled to `M25CSA032_prob4.pdf` using `pdflatex`.
- **How to run**:
  - Before running the classifier, install the required libraries (user‑level install):
    - `pip install --user scikit-learn pandas matplotlib seaborn`
  - Then run:
    - `python3 M25CSA032_prob4.py`
  - After execution, check the `nlp_final_results/` folder for CSV + figures and compile the LaTeX file to generate the final report PDF.

---

### Notes

- All Python files use only the libraries permitted in the assignment:
  - Problems 1–3: Python standard library only.
  - Problem 4: Scikit‑learn, NumPy, pandas, matplotlib, and seaborn for ML and plotting.
- File names and interfaces follow the exact conventions specified in `NLU_Assignment-1.pdf` so that each problem can be graded and executed independently.

