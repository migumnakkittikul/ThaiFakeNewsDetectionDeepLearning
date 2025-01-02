Thai Fake News Detection using Deep Learning
Overview
This project implements a Fake News Detection System for Thai-language news articles using a deep learning approach. It uses a Bi-LSTM with an Attention Mechanism and engineered features to classify news articles as Fake News or Fact News.

The system processes both the title and content of articles and includes advanced preprocessing for Thai text, such as tokenization, removal of stopwords, and custom feature engineering.

Features
Deep Learning Model: Bi-LSTM with Attention for contextual understanding of text.
Thai Text Preprocessing: Tokenization and cleaning using PyThaiNLP.
Engineered Features: Adds numerical features (e.g., Thai character ratio, number of URLs).
Augmentation: Augments data for the Fake News class using word-dropping and shuffling techniques.
Interactive Testing: Test the model with custom title and content.
Project Structure
bash
Copy code
.
├── main_notebook.ipynb      # The Jupyter Notebook containing all training and testing code.
├── tokenizer.pickle         # Saved tokenizer after training (required for inference).
├── best_model_checkpoint.pth # Model weights checkpoint (required for inference).
├── README.md                # Project documentation (this file).
├── Limesoda.jsonl           # Dataset (in JSONL format) for training and evaluation.
