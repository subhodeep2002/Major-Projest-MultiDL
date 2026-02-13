# MultiDL Hybrid Framework for Hindi News Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A hybrid deep learning framework designed to classify Hindi news articles with high accuracy by leveraging the combined strengths of traditional Machine Learning and modern Deep Learning models. This project addresses the scarcity of effective NLP models for low-resource languages like Hindi, providing a robust pipeline for data preprocessing, feature extraction, and classification.

The system benchmarks over seven standalone models and engineers multi-layer hybrid architectures to achieve significant performance gains over baseline methods.

---
## Key Features

* **Systematic Benchmarking:** Built and evaluated over 7 standalone models, including Naïve Bayes, Logistic Regression, CNN, BiLSTM, and BERT, to establish strong performance baselines.
* **Hybrid Model Engineering:** Designed and implemented advanced 2-layer and 3-layer hybrid models (e.g., **CNN-LR-SVM** and **SVM-BERT-LR**) to combine the unique strengths of different algorithms.
* **Significant Performance Lift:** The hybrid models improved classification **precision by up to 40%** and **F1-score by over 30%** compared to standalone baseline models.
* **Robustness on Imbalanced Data:** Enhanced classification reliability by **35%** across challenging and imbalanced news categories like crime, politics, and science.
* **Comprehensive NLP Pipeline:** Includes a full pipeline for data collection, cleaning, manual tagging (~3,000 articles into 10+ categories), and preprocessing for the Hindi language.

---
## Project Architecture

The core of this project is the development of two novel, three-layer hybrid models that strategically combine feature extraction and classification.



### 1. HLM-CLS (CNN-LR-SVM) Model

This model combines the statistical power of TF-IDF with the deep feature extraction of a CNN.

* **Layer 1: Feature Extraction:**
    * **TF-IDF Features:** A `TfidfVectorizer` captures the statistical importance of words.
    * **CNN Features:** A Convolutional Neural Network extracts higher-level patterns and semantic context.
* **Layer 2: Feature Reduction:**
    * The combined TF-IDF and CNN features are fed into parallel **Logistic Regression (LR)** and **Support Vector Machine (SVM)** models.
    * The probability outputs from these models are used as a new, reduced feature set.
* **Layer 3: Final Classification:**
    * A final **SVM model** is trained on the reduced feature set to make the final prediction.

### 2. HLM-SBL (SVM-BERT-LR) Model

This model leverages the powerful contextual embeddings from BERT.

* **Layer 1: Feature Extraction:**
    * **BERT Features:** A pre-trained BERT model generates deep, contextualized embeddings.
    * **TF-IDF & SVM Features:** TF-IDF vectors are generated and refined by an initial SVM layer.
* **Layer 2: Feature Combination:**
    * The features from the BERT and the initial SVM layer are combined.
* **Layer 3: Final Classification:**
    * A final **Logistic Regression model** is trained on the powerful combined feature set.

---
## Performance & Results

A comparative analysis showed that the hybrid models consistently outperformed standalone approaches. The two best-performing models were **CNN-LR-SVM** and **SVM-BERT-LR**.

**CLS Model Family Performance (Multi-Class)**
| Hybrid Model | Test Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **CNN-LR-SVM** | **61.11%** | **63.68%** | **61.11%** | **61.28%** |
| SVM-CNN-LR | 60.66% | 62.40% | 60.66% | 60.07% |

**SBL Model Family Performance (Multi-Class)**
| Hybrid Model | Test Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **SVM-BERT-LR** | **61.11%** | **62.17%** | **61.11%** | **60.87%** |
| LR-SVM-BERT | 60.96% | 61.85% | 60.96% | 60.96% |

---
## Tech Stack

* **Language:** Python 3.9+
* **Core Libraries:**
    * TensorFlow & Keras
    * Scikit-learn
    * Pandas
    * NumPy
    * NLTK
* **Environment:** Google Colab

---
## Future Work

* **Hinglish Support:** Develop models capable of classifying mixed Hindi-English (Hinglish) text.
* **Fine-Grained Classification:** Expand the model to classify news into more specific subcategories (e.g., cricket, football, or elections within sports and politics).
* **Personalized News Delivery:** Use the classification system as a backbone for a personalized news recommendation engine.

---
## Citation

This project is based on the Bachelor of Technology thesis submitted to the Bhilai Institute of Technology, Durg.

**Author:** Subhodeep Sarkar, Dr. K. Subhashini Spurjeon, Devdeep Sarkar, Niladri Ghosh, Aarushi Shrivastava..

---
## ©️ Copyright & Usage

This project is for demonstration purposes only. The code is proprietary and may not be used, copied, modified, or distributed without the express written permission of the author.
