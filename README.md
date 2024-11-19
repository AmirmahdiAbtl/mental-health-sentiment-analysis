# 🧠 Mental Health Sentiment Analysis  

This project implements a **mental health sentiment analysis system** using three different models: **Multinomial Naive Bayes (ML)**, **Bidirectional LSTM (RNN)**, and **RoBERTa (Transformer)**. The goal is to classify text data related to mental health issues (e.g., depression, anxiety, etc.) to support mental health initiatives.

---

## 🗂️ Project Highlights  

- **Dataset**: Utilized a diverse mental health dataset, categorized into 10 mental health-related topics.  
- **Model Types**: Applied three models:  
  - **Multinomial Naive Bayes** (Machine Learning)  
  - **Bidirectional LSTM** (Recurrent Neural Network)  
  - **RoBERTa** (Transformer-based model)  
- **Evaluation**: Measured model performance using accuracy, precision, recall, and F1-score.  

---
## 🗃️ Data

The dataset used for this project is available from [Zenodo](https://zenodo.org/records/3941387). It contains text data from online platforms such as Reddit, categorized into multiple mental health topics. These categories include:

- Depression  
- Anxiety  
- Suicide watch  
- ADHD  
- Autism  
- Addiction  
- Schizophrenia  
- Alcoholism  
- Loneliness  
- Healthy  

The data is preprocessed, with cleaning tasks such as removal of stopwords, stemming, and text normalization applied before model training.

---

## 1. **Multinomial Naive Bayes (Machine Learning)**

- **Overview**: A probabilistic classifier based on Bayes' Theorem, assuming independence between features. This model is effective for text classification problems.
- **How it Works**: It calculates the probability of each class given the features (words), and selects the class with the highest probability. It’s particularly efficient when dealing with large text datasets because of its simplicity and speed.
- **Advantages**: Fast training, less computationally expensive, works well with large sparse datasets (e.g., text).


## 2. **Bidirectional LSTM (RNN)**

- **Overview**: A type of Recurrent Neural Network (RNN) that processes sequences in both forward and backward directions. It can capture long-term dependencies in sequential data, making it effective for text analysis.
- **How it Works**: LSTM units have memory cells that can remember information over long sequences. The bidirectional layer allows the model to consider both past (forward direction) and future (backward direction) context when making predictions.
- **Advantages**: Can capture context from both directions, making it more powerful for tasks like sentiment analysis or sequence labeling.


## 3. **RoBERTa (Transformer-based model)**

- **Overview**: A transformer-based model built on the architecture of BERT (Bidirectional Encoder Representations from Transformers), optimized for better performance in various NLP tasks.
- **How it Works**: RoBERTa uses attention mechanisms to weigh the importance of different words in a sequence, learning contextual relationships between them. It uses a massive pre-trained model that is fine-tuned on specific tasks.
- **Advantages**: Achieves state-of-the-art results on many NLP benchmarks, especially for tasks requiring deep semantic understanding like sentiment analysis.

---

## 🚀 Getting Started  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/AmirmahdiAbtl/mental-health-sentiment-analysis.git  
   cd mental-health-sentiment-analysis  
   ```  

2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Run the **training** and **evaluation** notebooks to see the results for each model.

---

## 🛠️ Tools & Technologies  

- **Frameworks**: TensorFlow, Keras, Hugging Face (Transformers)  
- **Languages**: Python  
- **Text Processing**: NLTK  
- **Model Evaluation**: Scikit-learn  
- **Visualization**: Matplotlib  

---

## 📬 Contact  

👤 **Author**: [Amirmahdi Aboutalebi](https://github.com/AmirmahdiAbtl)  
📧 **Email**: amir.abootalebi2001@gmail.com  

Feel free to contribute, explore, or raise issues!  

---  
