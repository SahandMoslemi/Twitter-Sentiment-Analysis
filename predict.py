import numpy as np
from collections import Counter
from preprocessing import *

def euclidean_distance(point1, point2):
    point1 = np.array(point1, dtype=float)
    point2 = np.array(point2, dtype=float)
    return np.sqrt(np.sum((point1 - point2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = []
            for idx, train_point in enumerate(self.X_train):
                dist = euclidean_distance(test_point, train_point)
                distances.append((dist, self.y_train[idx]))
            
            k_nearest = sorted(distances, key=lambda x: x[0])[:self.k]
            k_nearest_labels = [label for _, label in k_nearest]
            
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return predictions

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = sum(pred == true for pred, true in zip(predictions, y_test))
        accuracy = correct / len(y_test)
        loss = 1 - accuracy 
        return accuracy, loss

if __name__ == "__main__":
    # Learn Tokens
    corpora = pd.read_csv(os.path.join('corpora', 'Reviews.csv')).sample(n=10000, random_state=1)['Text']
    no_bpe_iterations = 32
    BPE = BytePairEncoding(no_bpe_iterations, corpora)
    BPE.learn()

    fasttext_model = FastText(embedding_dim=128, epochs=100)
    new_fasttext_model = fasttext_model.load_model("fasttext_1")

    training_df = (pd.read_csv(os.path.join('data', 'twitter_training.csv')).rename(columns={'Positive': 'sentiment', 'im getting on borderlands and i will murder you all ,': 'text'}).sample(n=500, random_state=2))
    label_mapping = {
        "Negative": 0,
        "Positive": 1,
        "Irrelevant": 2,
        "Neutral": 3
    }
    training_y = training_df["sentiment"].map(label_mapping)

    validation_df = (pd.read_csv(os.path.join('data', 'twitter_training.csv')).rename(columns={'Positive': 'sentiment', 'im getting on borderlands and i will murder you all ,': 'text'}).sample(n=500, random_state=2))
    label_mapping = {
        "Negative": 0,
        "Positive": 1,
        "Irrelevant": 2,
        "Neutral": 3
    }
    validation_y = training_df["sentiment"].map(label_mapping)

    knn = KNN(k=3)
    knn.fit(training_df, training_y)
    
    predictions = knn.predict(validation_df)
    print("Predictions:", predictions)
    
    accuracy, loss = knn.score(validation_df, validation_y)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Test Loss: {loss:.2f}")
