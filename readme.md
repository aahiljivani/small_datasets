# Small Datasets — Exploratory ML & DL Mini‑Projects

> Hands‑on, reproducible notebooks that demonstrate classic supervised‑learning workflows on *very* small tabular datasets. Perfect for showcasing end‑to‑end thinking without GPU‑hungry data volumes.

---

## 📂 What’s Inside

| Notebook | Task                                                            | Library Stack                                     |
| -------- | --------------------------------------------------------------- | ------------------------------------------------- |
| ``       | Multi‑class classification on the classic *Iris* flower dataset | `pandas`, `scikit‑learn`, `matplotlib`, `seaborn` |
| ``       | Fish‑species prediction on the Kaggle *Fish Market* dataset     | `pandas`, `PyTorch`, `scikit‑learn`, `matplotlib` |

> **Why two notebooks?** They contrast a *traditional* ML workflow (scikit‑learn pipelines) with a *deep‑learning* pipeline (custom PyTorch loop) on similarly small, tidy datasets—highlighting trade‑offs in complexity, extensibility, and performance.

---

## 🚀 Quick Start

```bash
# 1️⃣  Clone the repo
$ git clone https://github.com/aahiljivani/small_datasets.git && cd small_datasets

# 2️⃣  Create a fresh environment (conda, mamba or venv)
$ python -m venv .venv && source .venv/bin/activate

# 3️⃣  Install requirements (coming soon!)
$ pip install -r requirements.txt

# 4️⃣  Launch Jupyter & run either notebook
$ jupyter lab
```

*Feel like hacking right away?* Open the notebook in **Google Colab**—no local setup needed.

---

## 📝 Project Summaries & Improvement Roadmap

### 1. *Iris* Classification (`iris_analysis.ipynb`)

**Current highlights**

- EDA with pair‑plots & box‑plots
- Encoded labels, trained Decision Tree, Random Forest, and k‑NN models
- Achieved ≈ 97‑99 % accuracy with `RandomForestClassifier`

**Areas for improvement**

1. **Pipeline consistency** – wrap preprocessing + model in a single `Pipeline` for easier grid‑search.
2. **Hyper‑parameter tuning** – run `GridSearchCV` or `Optuna` for fair model comparison.
3. **Cross‑validation** – replace train/test split with stratified 5‑fold CV.
4. **Model explainability** – add `shap` or feature‑importance plots.

---

### 2. Fish‑Breed Classification with PyTorch (`Pytorch_simple_fish_breed_classification.ipynb`)

**Current highlights**

- Scaled 6 numeric features, built a 3‑layer MLP with `ReLU` activations.
- Tracked loss per epoch and generated a full `classification_report`.
- Demonstrated PyTorch basics (tensor ops, manual training loop).

**Areas for improvement**

1. **Class imbalance** – use `WeightedRandomSampler` or class‑weighted `CrossEntropyLoss` to rescue *Whitefish* precision/recall.
2. **Refactor into **``** & **`` – enables batching, shuffling, and cleaner code reuse.
3. **Validation split & early stopping** – prevent over‑fitting on the tiny set.
4. **Baselines & storytelling** – compare against logistic regression or gradient‑boosting to justify DL.
5. **Repo hygiene** – move model code to `src/`, keep notebooks for EDA/visualisation.

---

## 🏗️ Suggested Repository Structure (next iteration)

```
small_datasets/
├── data/                   # raw or processed CSVs (git‑ignored if large)
├── notebooks/
│   ├── iris_analysis.ipynb
│   └── fish_classification.ipynb
├── src/
│   ├── iris_pipeline.py
│   └── fish_torch_train.py
├── tests/                  # minimal pytest coverage
├── requirements.txt        # pinned versions
└── README.md
```

---

## 📈 TODO List (open issues welcome!)

-

---

## 🤝 Contributing

Fork, branch off `main`, open a PR, and fill out the template. Even typo fixes are appreciated!

---

## 📜 License

TBD — most likely MIT once requirements & CI are in place.

