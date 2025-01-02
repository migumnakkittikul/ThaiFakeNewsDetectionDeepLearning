# Thai Fake News Detection using Deep Learning

## Overview

This project implements a **Fake News Detection System** for Thai-language news articles using a deep learning approach. It uses a **Bi-LSTM with an Attention Mechanism** and engineered features to classify news articles as **Fake News** or **Fact News**.

The system processes both the **title** and **content** of articles and includes advanced preprocessing for Thai text, such as tokenization, removal of stopwords, and custom feature engineering.

---

## Features

- **Deep Learning Model**: Bi-LSTM with Attention for contextual understanding of text.
- **Thai Text Preprocessing**: Tokenization and cleaning using PyThaiNLP.
- **Engineered Features**: Adds numerical features (e.g., Thai character ratio, number of URLs).
- **Augmentation**: Augments data for the `Fake News` class using word-dropping and shuffling techniques.
- **Interactive Testing**: Test the model with custom title and content.

---

## Project Structure

```
.
├── main_notebook.ipynb      # The Jupyter Notebook containing all training and testing code.
├── tokenizer.pickle         # Saved tokenizer after training (required for inference).
├── best_model_checkpoint.pth # Model weights checkpoint (required for inference).
├── README.md                # Project documentation (this file).
├── Limesoda.jsonl           # Dataset (in JSONL format) for training and evaluation.
```

---

## Requirements

- Python 3.7+
- PyTorch
- PyThaiNLP
- TensorFlow/Keras (for tokenizer)
- Scikit-learn
- tqdm
- Matplotlib
- Seaborn

You can install the required libraries using:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset is in **JSONL format** (e.g., `Limesoda.jsonl`) and contains the following fields:

- **Title**: Title of the news article.
- **Detail**: Content of the news article.
- **Document Tag**: Labels indicating "Fake News" or "Fact News".

---

## Example Usage

### Input Data:
```python
title = "พบ บ่อน้ำ ศักดิ์สิทธิ์ ใต้ ต้น โพธิ์ สมัย ศรีสัชนาลัย 800 ปี"
content = """26 ส.ค. 61 ผู้สื่อข่าวรายงานว่า ที่บริเวณทุ่งหญ้าเลี้ยงสัตว์กลางเขา
หมู่ 5 บ้านแสนตอ ต.สารจิตร อ.ศรีสัชนาลัย จ.สุโขทัย มีบ่อน้ำเก่าแก่ยุคเมืองโบราณ..."""
```

### Output:
```
Prediction: Fact News
```

---

## Model Details

- **Architecture**:
  - Embedding Layer
  - Bi-LSTM (x2)
  - Attention Mechanism
  - Fully Connected Layers (with Engineered Features)
- **Features**:
  - Thai text ratio
  - Number of URLs
  - Total length of the article
  - Number of Thai words

---

## Preprocessing

- **Tokenization**: Thai text is tokenized using `PyThaiNLP`'s `newmm` engine.
- **Stopword Removal**: Removes common Thai stopwords.
- **Augmentation**: Generates additional training examples for the `Fake News` class by shuffling or dropping words.

---

## Testing Custom Data

You can test any news article with the `test_model` function. For example:

```python
title = "Breaking News: Virus spreads through airborne transmission!"
content = "Government officials have confirmed that the virus is airborne..."
test_model(title, content)
```

---

## Training Process

- **Data Augmentation**: Balances the dataset by augmenting `Fake News` samples.
- **Training**: Trains the model for 30 epochs with early stopping based on validation accuracy.
- **Evaluation**: Evaluates the model on a held-out test set (5% of the dataset).

---

## Results

- **Accuracy**: 93% on the test set.
- **Classification Report**:
  ```
              Precision    Recall    F1-score
  Fake News      0.91       0.97        0.94
  Fact News      0.96       0.88        0.92
  ```

---

