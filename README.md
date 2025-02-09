# Fake News Detection 📰

## Overview
This project aims to build a machine learning model that detects whether a given news article is real or fake. The dataset contains news articles labeled as **real (0)** or **fake (1)** based on their content, author, and title.

## Dataset Description
The dataset consists of the following columns:
- `id`: Unique identifier for a news article
- `title`: Title of the news article
- `author`: Author of the news article
- `text`: Content of the article (may be incomplete)
- `label`: Classification of the news article: **1 (Fake News), 0 (Real News)**

## Technologies Used
- Python 🐍
- Scikit-learn 🤖
- Pandas 🏗
- NumPy 🔢
- Natural Language Processing (NLP) 🗣

## Installation
To run this project, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model
1. Load and preprocess the dataset
2. Convert text data using **TF-IDF Vectorization**
3. Train a **Machine Learning Model** (e.g., Logistic Regression, Naive Bayes, or SVM)
4. Evaluate model performance using accuracy, precision, recall, and F1-score

## Making Predictions
```python
# Load trained model
import pickle
model = pickle.load(open('fake_news_model.pkl', 'rb'))

# Predict on new sample
X_new = X_test[0].reshape(1, -1)  # Ensure correct shape
prediction = model.predict(X_new)

if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')

# Compare with actual label
print("Actual Label:", Y_test[0])
```

## Future Improvements
- Use **Deep Learning** (LSTMs, Transformers) for better accuracy
- Collect a **larger dataset** for improved generalization
- Deploy the model using **Flask or FastAPI** for real-time predictions

## Contributors
- **Sumit Kumar Jaiswal** - Developer ✨

## License
This project is licensed under the **MIT License**.

Happy Coding! 🚀

