# WINE QUALITY PREDICTION
## 1-Introduce
I am a 3rd-year AI/ML student at Konya Technical University. This project was given to me in the *Applied Artificial Intelligence* course.
I am excited to start new projects.

## 2-Why This Project ?
This is my first official machine learning project. I chose wine datasets to practice data analysis and learn how a machine learning workflow works.
While doing this project my aims are to learn Git and GitHub, improve my Python skills, understand data preprocessing and gain experience.
I will document everything from A to Z, so this project is also training for me.
 
---
## 3 - Project Overview
 
**Goal:** predict wine quality (`points`) and related attributes (`country`, `variety`, `price`) from expert wine reviews.
 
| Item | Description |
|---|---|
| Data type | Text (`description`) + tabular (categorical and numeric) |
| Task types | Regression (`points`), multi-class classification (`points_class`, `country`, `variety`, `price_class`) |
| Labeled? | Yes, supervised learning |
| Dataset 1 | [Wine Reviews, Kaggle](https://www.kaggle.com/datasets/zynicide/wine-reviews) (`winemag-data-130k-v2.csv`, real Wine Enthusiast reviews, not AI-generated) |
| Dataset 2 | *(to be added)* |
 
### Base papers
1. Kim, B. (2022). *Predictions of wine ratings using natural language processing.* Issues in Information Systems, 23(3), 64-68. https://doi.org/10.48009/3_iis_2022_107
2. Robson, F., & Amdahl-Culleton, L. (2018). *Classy classification: Classifying and generating expert wine review.* Stanford CS224N report. https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6909240.pdf
---

## 4 - Project Structure
 
```
wine-quality-prediction/
├── dataset/
│   └── raw/Kaggle/            # original CSV files (never edited by hand)
├── data_processed/            # train / dev / test CSVs created by the notebook
├── notebooks/
│   └── wine_reviews_analysis.ipynb
├── requirements.txt
└── README.md
```
---
## 4 - Project Structure
 
```
wine-quality-prediction/
├── dataset/
│   └── raw/Kaggle/            # original CSV files (never edited by hand)
├── data_processed/            # train / dev / test CSVs created by the notebook
├── notebooks/
│   └── wine_reviews_analysis.ipynb
├── requirements.txt
└── README.md
```
 
> Adjust this tree to match your real folders.
 
---
 
## 5 - Pipeline (what the notebook does)
 
### Step 1: Data analysis
- Shape, column types, number of unique values per column
- Missing values per column (percentage and bar chart)
- Duplicate rows and duplicate reviews
- Label check: how many classes each target has, class sizes (min / median / max), unlabeled rows
- Text features: review length in characters and words
> The dataset is not an image dataset, so RGB/grayscale and image size do not apply. Review length and word frequency are used instead.
 
### Step 2: Visualization
- Distribution of `points` and `price` (log scale)
- Top countries, varieties and tasters
- Class imbalance (long-tail plot)
- Price vs. points relationship (Spearman correlation, hexbin plot)
- Points by country and by taster (boxplots)
- Review length and most frequent words
### Step 3: Preprocessing
 
| Step | Why |
|---|---|
| Remove duplicate rows and duplicate reviews | Avoid data leakage between train and test |
| Clean text (lowercase, remove punctuation and digits) | Smaller vocabulary |
| Drop `taster_twitter_handle`, `region_1`, `region_2`, `designation`, `winery`, `title` | Irrelevant, too sparse, or too many classes |
| Handle missing values (`price` flag and median, `taster_name` = "Unknown") | Keep as many rows as possible |
| Create targets: `points_class` (4 bins), `price_class` (K-Means, k=10) | Allow classification as well as regression |
| Kim (2022) subset: 12 countries, 26 varieties, 12 tasters | Reproduce the paper's setup |
| Stratified 70 / 10 / 20 split (train / dev / test) | Same ratio as Robson & Amdahl-Culleton |
| Fit median and K-Means on train only | Avoid leakage |
 
### Step 4: Baseline models
- TF-IDF (unigrams + bigrams) + **Ridge regression** → `points`
- TF-IDF + **Multinomial Naive Bayes** → `points_class`, `country`, `variety`
---
 
## 6 - Literature Summary
 
| | Kim (2022) | Robson & Amdahl-Culleton (2018) |
|---|---|---|
| Problem | Regression (`points`) | Classification + review generation |
| Text representation | TF-IDF | GloVe embeddings |
| Models | Decision Tree, Gradient Boosting, Random Forest | Naive Bayes, LSTM (separate and combined), LSTM language model |
| Metrics | R² | Accuracy, F1-weighted, F1-macro, perplexity |
| Key result | Text + price + variety + taster: R² = 0.5927 (Gradient Boosting) | `points` is the hardest target; LSTM only slightly beats Naive Bayes |
 
### Planned methods
 
| Stage | Method | Target |
|---|---|---|
| A | TF-IDF + Ridge / Linear Regression | `points` |
| B | TF-IDF + Random Forest / Gradient Boosting | `points` |
| C | TF-IDF + Naive Bayes | `points_class`, `country`, `variety` |
| D | GloVe + 2-layer LSTM / GRU | same targets |
| E | Fine-tuned Transformer (DistilBERT / RoBERTa) | compare with LSTM |
 
---
 
## 7 - Results
 
> Fill this in after running the notebook. I will not copy numbers from the papers here.
 
| Model | Target | Metric | Dev | Test |
|---|---|---|---|---|
| TF-IDF + Ridge | `points` | R² | | |
| TF-IDF + Ridge | `points` | MAE | | |
| TF-IDF + Naive Bayes | `points_class` | Accuracy / F1-W / F1-M | | |
| TF-IDF + Naive Bayes | `country` | Accuracy / F1-W / F1-M | | |
| TF-IDF + Naive Bayes | `variety` | Accuracy / F1-W / F1-M | | |
 
**My comments on the results:** *(write here)*
 
---
 
## 8 - How to Run
 
```bash
python -m venv venv
venv/Scripts/Activate          # Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter notebook
```
 
Libraries used: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `kagglehub`.
 
---
## 9 - What I Learned
 
*(Add a new note here after each step of the project. Keep it short.)*
 
### Python notes
```
python -m venv venv                         = create a new virtual environment
venv/Scripts/Activate                       = activate that environment
python --version                            = show your Python version
python -m pip --version                     = show your pip path in the environment
python -m pip install --upgrade pip         = upgrade pip
python -m pip install pandas                = install the pandas library
python -m pip freeze > requirements.txt     = save your libraries and versions into requirements.txt
python -m pip install -r requirements.txt   = install everything listed in requirements.txt
```
 
### Pandas notes
```python
pd.read_csv("../dataset/raw/Kaggle/winemag-data.csv")   # ".." means go back one folder
common = df1.columns.intersection(df2.columns)           # common columns between df1 and df2
df = pd.concat([df1[common], df2[common]], ignore_index=True)  # merge only the common columns
df = df.drop(columns=["column1", "column2"])             # delete columns
```
 
### Data analysis and preprocessing notes *(new, fill in)*
- `df.info()`, `df.isna().sum()`, `df.nunique()` = first look at any dataset: **I learned:** ...
- Why duplicates must be removed before splitting: **I learned:** ...
- Why fit imputers / K-Means on train only (data leakage): **I learned:** ...
- Stratified split, why class balance matters: **I learned:** ...
### NLP / modeling notes *(new, fill in)*
- TF-IDF: **I learned:** ...
- Regression vs. classification metrics (R², MAE, accuracy, F1-weighted vs. F1-macro): **I learned:** ...
- Naive Bayes smoothing (alpha): **I learned:** ...
### Git / GitHub
```
git init                              = initialize a new Git repository in the current directory
git add .                             = stage all changed files
git status                            = list all changed files
git commit -m "comment"               = commit staged files
git branch -M main                    = rename the current branch to main
git push                              = upload committed changes to the remote repo
git push -u origin main               = push main to origin and remember it for future pushes
git remote add origin GITHUB_REPO_URL = connect the local repo to a remote GitHub repo
git config --global user.name "GitHub_Nickname"      = set your name
git config --global user.email "GitHub_Email_Address" = set your email
git restore notebooks/hello.ipynb     = discard changes in the working directory
```
 
When everything is set up, these 3 commands are enough for daily work:
 
```
git add .  →  git commit -m "comment"  →  git push
```
 
---
 
## 10 - Next Steps
- [ ] Run the notebook and fill in the Results table
- [ ] Add Random Forest / Gradient Boosting (Stage B)
- [ ] Add LSTM with GloVe (Stage D)
- [ ] Try a Transformer (Stage E)
- [ ] Add Dataset 2 and repeat the same pipeline
- [ ] Write final comments and conclusions
## 11 - References
1. Kim, B. (2022). https://doi.org/10.48009/3_iis_2022_107
2. Robson, F., & Amdahl-Culleton, L. (2018). https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6909240.pdf
3. Wine Reviews dataset: https://www.kaggle.com/datasets/zynicide/wine-reviews
