import numpy as np
from tqdm import tqdm
import logging
from config import RANDOM_STATE, EMBEDDING_DIM
from preprocessing import BytePairEncoding, FastText
import pandas as pd
import os


class BinarySVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Convert labels to -1 and 1
        y_ = np.where(y == 1, 1, -1)

        # Gradient descent
        for epoch in range(self.n_iterations):
            total_loss = 0
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
                
                # Calculate hinge loss
                margin = y_[idx] * (np.dot(x_i, self.w) + self.b)
                loss = max(0, 1 - margin) + self.lambda_param * np.sum(self.w ** 2)
                total_loss += loss
            
            avg_loss = total_loss / n_samples
            self.losses.append(avg_loss)
            
            if epoch % 100 == 0:
                logging.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)


class MulticlassSVM:
    def __init__(self, learning_rate=0.1, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.classifiers = {}
        self.n_classes = None
        self.losses = []  # Store average loss across all binary classifiers

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        X = np.array([np.array(x) for x in X])  # Convert list of lists to numpy array
        
        # Initialize lists to store losses
        all_losses = []
        
        # Train binary SVM for each class (One-vs-Rest)
        for class_idx in range(self.n_classes):
            logging.info(f"Training SVM for class {class_idx}")
            # Create binary labels
            binary_y = (y == class_idx).astype(int)
            
            # Initialize and train binary SVM
            svm = BinarySVM(
                learning_rate=self.lr,
                lambda_param=self.lambda_param,
                n_iterations=self.n_iterations
            )
            svm.fit(X, binary_y)
            self.classifiers[class_idx] = svm
            
            # Store losses for this classifier
            if len(all_losses) < len(svm.losses):
                all_losses.extend([[] for _ in range(len(svm.losses) - len(all_losses))])
            for i, loss in enumerate(svm.losses):
                all_losses[i].append(loss)
        
        # Calculate average loss across all classifiers for each iteration
        self.losses = [np.mean(losses) for losses in all_losses]
        
        # Print final average loss
        logging.info(f"Final average loss across all classifiers: {self.losses[-1]:.4f}")

    def predict(self, X):
        X = np.array([np.array(x) for x in X])  # Convert list of lists to numpy array
        predictions = []
        
        for x in tqdm(X, desc="Predicting"):
            # Get scores from all classifiers
            scores = []
            for class_idx in range(self.n_classes):
                classifier = self.classifiers[class_idx]
                score = np.dot(x, classifier.w) + classifier.b
                scores.append(score)
            
            # Predict the class with highest score
            predicted_class = np.argmax(scores)
            predictions.append(predicted_class)
            
        return np.array(predictions)

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = sum(pred == true for pred, true in zip(predictions, y_test))
        accuracy = correct / len(y_test)
        loss = 1 - accuracy
        return accuracy, loss
    

if __name__ == "__main__":
    # Example usage remains the same as in your original code
    # Training
    logging.info("Loading BPE Model ...")
    new_BPE = BytePairEncoding()
    new_BPE.load(os.path.join("models", f"bpe-2048"))

    # Load FastText Model
    logging.info("Loading FastText Model ...")
    fasttext_model = FastText(embedding_dim=EMBEDDING_DIM)
    new_fasttext_model = fasttext_model.load_model(os.path.join("models", f"fasttext-300-40"))

    # Load and prepare training data
    training_df = pd.read_csv(
        os.path.join("data", "twitter_training.csv"),
        names=["id", "movie", "sentiment", "text"],
        header=0
    ).sample(n=500, random_state=RANDOM_STATE)

    # Data preprocessing steps...
    min_count = training_df["sentiment"].value_counts().min()
    training_df = training_df.groupby("sentiment").apply(
        lambda x: x.sample(min_count, random_state=RANDOM_STATE)
    ).reset_index(drop=True)

    # Prepare features and labels
    training_df["text_tokenized"] = training_df["text"].apply(new_BPE.tokenize)
    training_df["text_avg_embeddings"] = training_df["text_tokenized"].apply(
        new_fasttext_model.get_average_embedding
    )

    label_mapping = {
        "Negative": 0,
        "Positive": 1,
        "Irrelevant": 2,
        "Neutral": 3
    }
    training_df["int_labels"] = training_df["sentiment"].map(label_mapping)

    # Train model
    learning_rate = 0.1
    lambda_param = 0.01
    n_iterations = 1000
    logging.info("Training SVM Model...")
    svm = MulticlassSVM(learning_rate=learning_rate, lambda_param=lambda_param, n_iterations=n_iterations)
    svm.fit(training_df["text_avg_embeddings"], training_df["int_labels"])

    # Validation
    validation_df = pd.read_csv(
        os.path.join("data", "twitter_validation.csv"),
        names=["id", "movie", "sentiment", "text"],
        header=0
    ).sample(n=100, random_state=RANDOM_STATE)

    # Prepare validation data
    min_count = validation_df["sentiment"].value_counts().min()
    validation_df = validation_df.groupby("sentiment").apply(
        lambda x: x.sample(min_count, random_state=RANDOM_STATE)
    ).reset_index(drop=True)

    validation_df["text_tokenized"] = validation_df["text"].apply(new_BPE.tokenize)
    validation_df["text_avg_embeddings"] = validation_df["text_tokenized"].apply(
        new_fasttext_model.get_average_embedding
    )
    validation_df["int_labels"] = validation_df["sentiment"].map(label_mapping)

    # Evaluate model
    logging.info("Evaluating SVM Model...")
    svm_accuracy, svm_loss = svm.score(
        validation_df["text_avg_embeddings"], 
        validation_df["int_labels"]
    )
    logging.info(f"SVM Accuracy: {svm_accuracy}")
    logging.info(f"SVM Loss: {svm_loss}")