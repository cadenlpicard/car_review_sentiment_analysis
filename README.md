

This project is a sentiment analysis model that classifies car reviews into three categories: Negative (0), Neutral (1), and Positive (2). The model uses a deep learning approach with an LSTM-based neural network to classify the sentiment of car reviews. The data is processed in Google Colab with GPU support, which helps improve model training speed.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [File Outputs](#file-outputs)

## Project Structure

```
.
├── data/
│   ├── Train_data.xlsx              # Training data with "Review" and "Target" columns
│   ├── Test_features.xlsx           # Test data with only "Review" column
├── predictions.csv                  # Output file containing predictions on the test set
└── README.md                        # Project documentation
```

## Dataset

The dataset consists of car reviews, each labeled with one of three sentiment classes:
- **0** - Negative
- **1** - Neutral
- **2** - Positive

### Training Data

- **Train_data.xlsx**: Contains car reviews and corresponding sentiment labels in the `Target` column.

### Test Data

- **Test_features.xlsx**: Contains car reviews without labels in the `Review` column. The model will predict the sentiment for these reviews.

## Requirements

The project uses Python 3 and the following libraries:

- `pandas`
- `numpy`
- `tensorflow`
- `scikit-learn`

You can install these packages using:

```bash
pip install pandas numpy tensorflow scikit-learn
```

## Model Architecture

The model is a neural network with the following layers:

1. **Embedding Layer**: Converts words into dense vectors of fixed size.
2. **Bidirectional LSTM Layer**: Allows the model to learn patterns from both directions in a sequence, helping it capture context.
3. **Dropout Layers**: Help prevent overfitting by randomly setting inputs to zero during training.
4. **Dense Layers**: Used for classification, with a softmax activation for multi-class output.

The model is compiled with `categorical_crossentropy` loss for multi-class classification and `adam` optimizer.

## Usage

### 1. Load the Data

Place your training data (`Train_data.xlsx`) and test data (`Test_features.xlsx`) in the appropriate directory.

### 2. Run the Script

Execute the script in a Python environment. You can use Google Colab for GPU support to speed up training.

### 3. Train the Model

The script will:
- Preprocess the data by tokenizing and padding the text reviews.
- One-hot encode the labels for multi-class classification.
- Use `class_weight` to handle any class imbalance in the training data.
- Train the model for 10 epochs and validate on a separate validation set.

### 4. Predict on Test Data

The model will make predictions on the test set and save them in `predictions.csv`.

### 5. Check Predictions

The output file, `predictions.csv`, contains the original review and the predicted sentiment label.

## Evaluation

The model’s performance is evaluated on a validation set with the following metrics:
- **Accuracy**: Measures the percentage of correct predictions.
- **Precision, Recall, F1-Score**: Evaluates the model's performance for each sentiment class (Negative, Neutral, Positive).

The model's evaluation is printed to the console after training on the validation set.

### Sample Metrics (Validation Set)

```
Validation Accuracy: 0.82
Classification Report on Validation Set:
              precision    recall  f1-score   support

    Negative       0.80      0.78      0.79
     Neutral       0.83      0.85      0.84
    Positive       0.85      0.83      0.84

   micro avg       0.82      0.82      0.82
   macro avg       0.83      0.82      0.82
weighted avg       0.82      0.82      0.82
```

## File Outputs

- **`predictions.csv`**: Contains test reviews and predicted sentiment labels. Each row includes:
  - `Review`: The review text.
  - `Predicted Label`: The sentiment prediction (`0` for Negative, `1` for Neutral, and `2` for Positive).

---

This project provides a framework for classifying sentiment in text data using deep learning. With minor modifications, the model can be adapted for similar multi-class classification tasks.
