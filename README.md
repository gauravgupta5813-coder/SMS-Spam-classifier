# SMS Spam Classifier with Imbalance Handling (SMOTE)

An end-to-end Machine Learning pipeline utilizing Natural Language Processing (NLP) to automate the classification of Short Message Service (SMS) text messages into operational designations: **Ham** (legitimate communications) and **Spam** (unsolicited or malicious traffic). 

This project specifically highlights data preparation configurations, mathematical feature space adjustments, and structural class asymmetry mitigation using synthetic data oversampling techniques.

---

## 📈 Executive Performance Summary
Standard accuracy scores can become deceptive metrics when working on heavily unbalanced datasets. This model heavily prioritizes capturing genuine text alerts safely (**High Ham Precision**) while minimizing overlooked security issues (**High Spam Recall**):

* **Global Classification Accuracy:** `96.68%`
* **Ham Classification (Class 0):** `0.99 Precision` | `0.97 Recall` | `0.98 F1-Score`
* **Spam Classification (Class 1):** `0.83 Precision` | `0.94 Recall` | `0.88 F1-Score`

---

## 🛠️ Operational Workflow & Core Architecture

The system transitions raw, noisy data strings into clear classification outputs through a structured, sequential development pipeline:

### 1. Data Parsing & Universal Normalization
* **Structural Optimization:** Unindexed sparse data columns (`Unnamed: 2`, `3`, and `4`) are dropped to save memory space.
* **Binary Target Mapping:** Categorical status text labels (`ham` / `spam`) are converted to strict numeric types (`0` and `1`).

### 2. Specialized Text Cleaning Pipeline
To strip vocabulary variations and linguistic noise, text variables undergo four sequential cleaning steps:
* **Universal Case Forcing:** Standardizes all characters to lower case so variations like `FREE`, `Free`, and `free` map identically.
* **Noise Filtering:** Strips punctuation matrices and localized unique special characters using translation parameters.
* **Stopword Elimination:** Leverages standard NLTK English corpora filters to drop highly recurrent, low-semantic tokens (e.g., *'the'*, *'is'*, *'at'*).
* **Morphological Stemming:** Applies the Porter Stemming algorithm to recursively trim structural word suffixes back to their core base formats (e.g., *'companies'* transforms to *'comp'*, *'joking'* maps to *'joke'*).

### 3. Feature Matrix Construction (TF-IDF Vectorization)
Tokens are transformed into vectorized mathematical dimensions using Term Frequency-Inverse Document Frequency (**TF-IDF**) vector scoring. This scales unique diagnostic keyword values up while damping generic words that frequently repeat throughout the dataset.

### 4. Synthetic Class Balancing (SMOTE Implementation)
The raw data distributions showed an acute class asymmetry problem:
* **Ham (Class 0):** 4,825 instances
* **Spam (Class 1):** 747 instances

Left untreated, the training phase would heavily favor predicting "Ham". To address this safely, the pipeline implements **SMOTE** (Synthetic Minority Over-sampling Technique) strictly *after* splitting the data into separate training and testing blocks. This avoids data leakage into validation spaces while synthesizing coherent minority data vectors to generate perfectly balanced 50/50 test sets:
* **Balanced Training Split Class 0:** 3,860 samples
* **Balanced Training Split Class 1:** 3,860 samples

### 5. Probabilistic Classifier Fit
A **Multinomial Naive Bayes (MultinomialNB)** algorithm is trained on the resampled vector spaces. This technique is highly optimal for handling large, sparse text frequency matrices with low computational overhead.

---

## 🔬 Validation & Live Inference Verification

To simulate true live deployment settings, the pipeline features an operational prediction function that handles raw strings from scratch:

* **Sample Inference Input:** `"Your Amazon order has been delivered"`
* **Output Vector Classification Target:** `[0] -> (Not Spam)`

🚀 Future Development Goals
Hyperparameter Tuning: Implement automated cross-validation grid searches (GridSearchCV) to tune TfidfVectorizer n-gram ranges and Naive Bayes smoothing variables (alpha).

Algorithmic Benchmarking: Test performance curves against sequential baseline structures including Logistic Regression, Support Vector Machines (SVM), and Random Forests.

Model Deployment Persistence: Integrate native serialization routines utilizing joblib or pickle to export pipeline binaries, serving as a backend for standalone API architectures.

👥 Authorship & Acknowledgements
Gaurav Gupta - Data Science Portfolio Framework Execution
---

## 📂 Project Repository Structure
```text
├── data/
│   └── spam.csv                  # Raw dataset containing text corpus sequences
├── notebooks/
│   └── Spam_Classifier.ipynb     # Complete documented execution notebook 
└── README.md                     # Portfolio documentation and project index
|__ requirements.txt              # List of libraries (pandas, nltk, scikit-learn, imblearn



