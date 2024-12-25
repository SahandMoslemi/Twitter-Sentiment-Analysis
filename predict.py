from preprocessing import BytePairEncoding, FastText, GoogleNewsEmbedding
from config import NO_BPE_ITERATIONS
import numpy as np
from collections import Counter
from preprocessing import *
from tqdm import tqdm


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
        for test_point in tqdm(X_test, desc="Predicting"):
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
        logging.info(predictions)
        correct = sum(pred == true for pred, true in zip(predictions, y_test))
        accuracy = correct / len(y_test)
        loss = 1 - accuracy 
        return accuracy, loss

if __name__ == "__main__":
    k_values = [1, 3, 5, 7, 9]
    results = []

    # Training
    # Load BPE Model
    logging.info("Loading BPE Model ...")
    new_BPE = BytePairEncoding()
    new_BPE.load(os.path.join("models", f"bpe-2048"))

    # Load FastText Model
    logging.info("Loading FastText Model ...")
    fasttext_model = FastText(embedding_dim=300)
    new_fasttext_model = fasttext_model.load_model(os.path.join("models", f"fasttext-300-40"))

    # Load Training Dataset
    training_df = pd.read_csv(
        os.path.join("data", "twitter_training.csv"),
        names=["id", "movie", "sentiment", "text"],
        header=0
    ).sample(n=50000, random_state=RANDOM_STATE)

    logging.info("Downsampling the dataset ...")

    # Ensure equal number of samples for each sentiment category
    min_count = training_df["sentiment"].value_counts().min()
    training_df = training_df.groupby("sentiment").apply(lambda x: x.sample(min_count, random_state=RANDOM_STATE)).reset_index(drop=True)


    logging.info(f"Training Dataset: {training_df.head()}, Sentiment Distribution: {training_df['sentiment'].value_counts()}")

    # Tokenize the training dataset
    training_df["text_tokenized"] = training_df["text"].apply(new_BPE.tokenize)

    # Get the embeddings
    training_df["text_avg_embeddings"] = training_df["text_tokenized"].apply(new_fasttext_model.get_average_embedding)

    # Map the labels to integers
    label_mapping = {
        "Negative": 0,
        "Positive": 1,
        "Irrelevant": 2,
        "Neutral": 3
    }
    training_df = training_df[training_df["sentiment"] != "Irrelevant"]
    training_df["int_labels"] = training_df["sentiment"].map(label_mapping)

    logging.info("Training the KNN Model ...")

    # Validation
    # Load Validation Dataset
    validation_df = pd.read_csv(
        os.path.join("data", "twitter_validation.csv"),
        names=["id", "movie", "sentiment", "text"],
        header=0
    ).sample(n=700, random_state=RANDOM_STATE)

    logging.info("Downsampling the validation dataset ...")

    # Ensure equal number of samples for each sentiment category
    min_count = validation_df["sentiment"].value_counts().min()
    validation_df = validation_df.groupby("sentiment").apply(lambda x: x.sample(min_count, random_state=RANDOM_STATE)).reset_index(drop=True)


    logging.info(f"Validation Dataset: {validation_df.head()}, Sentiment Distribution: {validation_df['sentiment'].value_counts()}")

    # Tokenize the validation dataset
    validation_df["text_tokenized"] = validation_df["text"].progress_apply(new_BPE.tokenize)

    # Get the embeddings
    validation_df["text_avg_embeddings"] = validation_df["text_tokenized"].progress_apply(new_fasttext_model.get_average_embedding)

    # Map the labels to integers
    validation_df["int_labels"] = validation_df["sentiment"].map(label_mapping)

    logging.info("Evaluating the KNN Model ...")

    for k in k_values:
        logging.info(f"Training and evaluating KNN Model with k={k} ...")
        knn = KNN(k=k)
        knn.fit(training_df["text_avg_embeddings"], training_df["int_labels"])
        accuracy, loss = knn.score(validation_df["text_avg_embeddings"], validation_df["int_labels"])
        results.append((k, accuracy, loss))

    # Google News Embeddings
    # Get the embeddings using Google News Word2Vec model
    logging.info("Getting embeddings using Google News Word2Vec model ...")
    google_news_model_path = os.path.join("models", "GoogleNews-vectors-negative300.bin.gz")
    google_news_embedding = GoogleNewsEmbedding(google_news_model_path)
    validation_df["google_news_avg_embeddings"] = validation_df["text_tokenized"].apply(google_news_embedding.get_average_embedding)

    for k in k_values:
        logging.info(f"Training and evaluating KNN Model with Google News embeddings and k={k} ...")
        knn_google_news = KNN(k=k)
        knn_google_news.fit(training_df["text_avg_embeddings"], training_df["int_labels"])
        accuracy_google_news, loss_google_news = knn_google_news.score(validation_df["google_news_avg_embeddings"], validation_df["int_labels"])
        results.append((k, accuracy_google_news, loss_google_news))

    for k, accuracy, loss in results:
        logging.info(f"k={k} - Accuracy: {accuracy:.2f}, Test Loss: {loss:.2f}")
