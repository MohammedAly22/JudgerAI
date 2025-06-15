# 🏩 JudgerAI — Your Dream Legal AI Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/MohammedAly22/JudgerAI.svg)](https://github.com/MohammedAly22/JudgerAI/stargazers)
[![Forks](https://img.shields.io/github/forks/MohammedAly22/JudgerAI.svg)](https://github.com/MohammedAly22/JudgerAI/network)

![A virtual avatar for a legal judge called _JudgerAI_ sitting on a coury, the avatr needs to be stunning and catchy  The name _JudgerAI_ needs to appear in the image (2)](https://github.com/user-attachments/assets/dc466ebe-aec7-4a9e-b987-c43a5caa0e48)
---

**Table of Contents:**
* [🚀 Overview](#-overview)
* [📺 Demo](#-demo)
* [⚙️ Installation](#%EF%B8%8F-installation)
* [📊 Dataset](#-dataset)
* [🧹 Project Modules](#-project-modules)
* [🤖 AI Models](#-ai-models)
* [🧪 Experiments & Evaluation](#-experiments--evaluation)
* [🏅 Training Approach](#-training-approach)
* [📊 Ensemble Voting](#-ensemble-voting)
* [📚 Notebooks & Further Reading](#-notebooks--further-reading)

## 🚀 Overview

**JudgerAI** is an innovative NLP application that **predicts legal case outcomes** with impressive accuracy by analyzing past cases, precedents, and case facts. It empowers legal professionals to:

* 📈 Increase prediction accuracy
* ⏱️ Save valuable time on case research
* 🧠 Make informed, data-driven decisions

---

## 📺 Demo

Watch **JudgerAI** in action:

[![JudgerAI Demo](https://github.com/MohammedAly22/JudgerAI/assets/90681796/b5702afd-64f7-4d17-b4bd-c1ad57262260)](https://github.com/MohammedAly22/JudgerAI/assets/90681796/b5702afd-64f7-4d17-b4bd-c1ad57262260)

---

## ⚙️ Installation

1️⃣ **Clone the repository**

   ```bash
   git clone https://github.com/MohammedAly22/JudgerAI.git
   ```

2️⃣ **Download GloVe embeddings** (50‌-dim) from Kaggle and save as:

   ```
   ./GloVe/glove.6B.50d.txt
   ```

3️⃣ **Download pre-trained models** (heavy files) and place them in `models/`.

4️⃣ **Directory structure**:

   ```text
   JudgerAI/
   ├── csvs/
   ├── dataset/
   ├── GloVe/
   ├── models/
   ├── src/
   └── *.ipynb
   ```

5️⃣ **Run the app**:

   ```bash
   streamlit run src/main.py
   ```
---

## 📊 Dataset

* **Total cases:** 3,464
* **Key columns:** ID, name, href, first/second\_party, winning\_party, winner\_index (0/1), facts
* **Input:** `facts` → **Output:** `winner_index`

Here is the dataset summary:
  
| column          | datatype  | description                                                                                |
| ---             | ---       | ---                                                                                        |
| `ID`            | int64     | Defines the case ID                                                                        |
| `name`          | string    | Defines the case name                                                                      |
| `href`          | string    | Defines the case hyper-reference                                                           |
| `first_party`   | string    | Defines the name of the first party (petitioner) of a case                                 |
| `second_party`  | string    | Defines the name of the second party (respondent) of a case                                |
| `winning_party` | string    | Defines the winning party name of a case                                                   |
| `winner_index`  | int64     | Defines the winning index of a case, 0 => the first party wins, 1 => the second party wins |
| `facts`         | string    | Contains the case facts that are needed to determine who is the winner of a specific case  |

---

## 🧹 Project Modules

Modular structure for maintainability and clarity:

| Module               | Location                  | Responsibilities                                                    |
| -------------------- | ------------------------- | ------------------------------------------------------------------- |
| **Preprocessing**    | `src/preprocessing.py`    | Tokenization, balancing, anonymization, vectorization               |
| **Plotting**         | `src/plotting.py`         | Visualizing performance, confusion matrices, ROC-AUC, heatmaps      |
| **Utils**            | `src/utils.py`            | Training helpers, k-fold CV, accuracy/loss summary builders         |
| **Streamlit App**    | `src/main.py`             | Frontend UI for demo and deployment                                 |
| **Deployment Utils** | `src/deployment_utils.py` | Model loader, sample picker, vectorizer generator, highlights words |

---

## 🤖 AI Models

JudgerAI incorporates **7** different models:

* **Doc2Vec** – Documents as dense vectors
* **1D-CNN** – Convolutional features over text
* **TF-IDF + TextVectorization** – Weighted bag-of-words embedding
* **GloVe** – Global co-occurrence embeddings
* **FastText** – Subword-enhanced embeddings
* **LSTM** – Memory‌-capable sequences
* **BERT** – Contextual pre-trained transformer

A mix of traditional and modern architectures to maximize coverage.

---

## 🧪 Experiments & Evaluation

Three core preprocessing decisions were evaluated:

1. **Preprocessing steps** – stopword removal, stemming, etc.
2. **Data anonymization** – replacing party names with `_PARTY_`
3. **Label imbalance** – strategies for balanced classes

This results in **2³ = 8 experiments**, each run with **4-fold cross-validation**, giving thorough analysis across **32 total runs per model**.

|Combination # | Preprocessing | Data Anonymization | Label Class Imbalance |
|---           | ---           | ---                | ---                   |
|  1           |   No          |  No                |  No                   |
|  2           |   No          |  No                |  Yes                  |
|  3           |   No          |  Yes               |  No                   |
|  4           |   No          |  Yes               |  Yes                  |
|  5           |   Yes         |  No                |  No                   |
|  6           |   Yes         |  No                |  Yes                  |
|  7           |   Yes         |  Yes               |  No                   |
|  8           |   Yes         |  Yes               |  Yes                  |
---

## 🏅 Training Approach
![graph](https://github.com/MohammedAly22/JudgerAI/assets/90681796/f8e1fd26-8697-45d1-8cb3-439a7b14dc6c)

* **80/20** train/test split
* **4-fold CV** on training set
* Best combination selected per model based on accuracy

**Workflow:**

* Train each model × 8 preprocessing setups × 4 CV folds
* Evaluate and select **best-performing model**

---

## 📊 Ensemble Voting

Final predictions are generated through an **ensemble voting** method across all tuned models to ensure robustness.

---

## 📚 Notebooks & Further Reading

In-depth exploration available in the following notebooks:

* `BERT_experiments.ipynb`
* `cnn_experiments.ipynb`
* `doc2vec_experiments.ipynb`
* `FastText_experiments.ipynb`
* `glove_experiments.ipynb`
* `LSTM_experiments.ipynb`
* `tf_idf_experiments.ipynb`
* `voting_experiments.ipynb`

---

## 🤝 Contributing & License

Contributions welcome! Feel free to contribute to JudgerAI.

---

## 🙏 Thank You!

Thank you for exploring **JudgerAI**—ushering in a smarter future for legal decision-making with AI-powered precision.

---
