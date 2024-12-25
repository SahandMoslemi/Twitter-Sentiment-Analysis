import numpy as np
from preprocessing import BytePairEncoding
from config import NO_BPE_ITERATIONS
import numpy as np
from preprocessing import *


import numpy as np

class RNN:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.W_recurrent = [
            np.random.randn(hidden_sizes[0], hidden_sizes[0]) * 0.01,
            np.random.randn(hidden_sizes[1], hidden_sizes[1]) * 0.01,
            np.random.randn(hidden_sizes[2], hidden_sizes[2]) * 0.01
        ]
        
        self.W_forward = [
            np.random.randn(hidden_sizes[0], input_size) * 0.01,      
            np.random.randn(hidden_sizes[1], hidden_sizes[0]) * 0.01, 
            np.random.randn(hidden_sizes[2], hidden_sizes[1]) * 0.01, 
            np.random.randn(output_size, hidden_sizes[2]) * 0.01      
        ]
        
        self.biases = [
            np.zeros((hidden_sizes[0], 1)),
            np.zeros((hidden_sizes[1], 1)),
            np.zeros((hidden_sizes[2], 1)),
            np.zeros((output_size, 1))
        ]

    def forward_step(self, x, hidden_states):
        new_hidden_states = []
        h = x.reshape(-1, 1)  
        
        for i in range(3): 
            h = np.tanh(
                np.dot(self.W_forward[i], h) +
                np.dot(self.W_recurrent[i], hidden_states[i]) +
                self.biases[i]
            )
            new_hidden_states.append(h)
        
        # Output layer
        output = np.dot(self.W_forward[3], h) + self.biases[3]
        
        return output, new_hidden_states

    def process_sequence(self, input_sequence):
        hidden_states = [
            np.zeros((self.hidden_sizes[0], 1)),
            np.zeros((self.hidden_sizes[1], 1)),
            np.zeros((self.hidden_sizes[2], 1))
        ]
        
        all_hidden_states = [[state] for state in hidden_states]
        all_inputs = []
        all_outputs = []
        
        for x in input_sequence:
            all_inputs.append(x)
            output, hidden_states = self.forward_step(x, hidden_states)
            all_outputs.append(output)
            
            for i, state in enumerate(hidden_states):
                all_hidden_states[i].append(state)
        
        final_output = all_outputs[-1]
        probabilities = self.softmax(final_output)
        
        return probabilities, all_hidden_states, all_inputs, all_outputs

    def backward_step(self, dh_next, hidden_states, prev_hidden_states, x, t):
        dW_forward = [np.zeros_like(w) for w in self.W_forward]
        dW_recurrent = [np.zeros_like(w) for w in self.W_recurrent]
        db = [np.zeros_like(b) for b in self.biases]
        
        dh = dh_next[-1]  
        
        for i in reversed(range(3)):
            h = hidden_states[i]
            h_prev = prev_hidden_states[i]
            
            dtanh = (1 - h * h) * dh
            
            if i == 0:
                dW_forward[i] += np.dot(dtanh, x.reshape(1, -1))
            else:
                dW_forward[i] += np.dot(dtanh, hidden_states[i-1].T)
            
            dW_recurrent[i] += np.dot(dtanh, h_prev.T)
            db[i] += dtanh
            
            if i > 0:
                dh = np.dot(self.W_forward[i].T, dtanh)
        
        return dW_forward, dW_recurrent, db

    def backprop_through_time(self, input_sequence, hidden_states, target, probabilities):
        T = len(input_sequence)
        
        dW_forward_total = [np.zeros_like(w) for w in self.W_forward]
        dW_recurrent_total = [np.zeros_like(w) for w in self.W_recurrent]
        db_total = [np.zeros_like(b) for b in self.biases]
        
        dout = probabilities.copy()
        dout[target] -= 1
        
        dW_forward_total[3] = np.dot(dout, hidden_states[2][-1].T)
        db_total[3] = dout
        
        dh_next = [np.dot(self.W_forward[3].T, dout)]
        
        # Backpropagate through time
        for t in reversed(range(T)):
            current_states = [states[t+1] for states in hidden_states]
            prev_states = [states[t] for states in hidden_states]
            
            dW_forward, dW_recurrent, db = self.backward_step(
                dh_next, current_states, prev_states, input_sequence[t], t
            )
            
            for i in range(3):
                dW_forward_total[i] += dW_forward[i]
                dW_recurrent_total[i] += dW_recurrent[i]
                db_total[i] += db[i]
            
            dh_next = [np.dot(self.W_recurrent[i].T, db[i]) for i in range(3)]
        
        return dW_forward_total, dW_recurrent_total, db_total

    def train_step(self, input_sequence, target):
        probabilities, hidden_states, inputs, _ = self.process_sequence(input_sequence)
        
        loss = -np.log(probabilities[target])
        
        dW_forward, dW_recurrent, db = self.backprop_through_time(
            inputs, hidden_states, target, probabilities
        )
        
        for i in range(len(self.W_forward)):
            self.W_forward[i] -= self.learning_rate * dW_forward[i]
            self.biases[i] -= self.learning_rate * db[i]
            
        for i in range(len(self.W_recurrent)):
            self.W_recurrent[i] -= self.learning_rate * dW_recurrent[i]
        
        return loss

    def predict(self, input_sequence):
        probabilities, _, _, _ = self.process_sequence(input_sequence)
        return np.argmax(probabilities)

    def fit(self, X, y, epochs=10, batch_size=32):
        n_samples = len(X)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                batch_loss = 0
                
                for idx in batch_indices:
                    loss = self.train_step(X[idx], y[idx])
                    batch_loss += loss
                
                epoch_loss += batch_loss / len(batch_indices)
            
            avg_loss = epoch_loss / (n_samples // batch_size)
            losses.append(avg_loss)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
        
        return losses

    def score(self, X_test, y_test):
        correct = 0
        total_loss = 0
        
        for inputs, target in zip(X_test, y_test):
            probabilities, _, _, _ = self.process_sequence(inputs)
            pred = np.argmax(probabilities)
            if pred == target:
                correct += 1
            total_loss -= np.log(probabilities[target])
        
        accuracy = correct / len(y_test)
        avg_loss = total_loss / len(y_test)
        
        return accuracy, avg_loss

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

if __name__ == "__main__":
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
    ).sample(n=3000, random_state=RANDOM_STATE)

    logging.info("Downsampling the dataset ...")

    # Ensure equal number of samples for each sentiment category
    min_count = training_df["sentiment"].value_counts().min()
    training_df = training_df.groupby("sentiment").apply(lambda x: x.sample(min_count, random_state=RANDOM_STATE)).reset_index(drop=True)

    logging.info(f"Training Dataset: {training_df.head()}, Sentiment Distribution: {training_df['sentiment'].value_counts()}")

    # Tokenize the training dataset
    training_df["text_tokenized"] = training_df["text"].apply(new_BPE.tokenize)

    # Map the labels to integers
    label_mapping = {
        "Negative": 0,
        "Positive": 1,
        "Irrelevant": 2,
        "Neutral": 3
    }
    training_df["int_labels"] = training_df["sentiment"].map(label_mapping)

    logging.info("Training the KNN Model ...")

    # Initialize RNN with 3 hidden layers
    input_size = 300  # FastText embedding dimension
    hidden_sizes = [256, 128, 64]  # Three hidden layers with decreasing sizes
    output_size = 4   # Number of sentiment classes
    rnn = RNN(input_size, hidden_sizes, output_size, learning_rate=0.1)

    training_df["sequence_embeddings"] = training_df["text_tokenized"].apply(new_fasttext_model.get_embeddings)

    # Train RNN
    logging.info("Training RNN model...")
    X_train = training_df["sequence_embeddings"].values
    y_train = training_df["int_labels"].values
    
    epochs=10
    batch_size=32
    losses = rnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

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
    validation_df["text_tokenized"] = validation_df["text"].apply(new_BPE.tokenize)

    # Map the labels to integers
    validation_df["int_labels"] = validation_df["sentiment"].map(label_mapping)

    # Evaluate RNN
    validation_df["sequence_embeddings"] = validation_df["text_tokenized"].apply(new_fasttext_model.get_embeddings)
    logging.info("Evaluating RNN model...")
    X_val = validation_df["sequence_embeddings"].values
    y_val = validation_df["int_labels"].values
    
    accuracy, loss = rnn.score(X_val, y_val)
    logging.info(f"RNN Accuracy: {accuracy}")
    logging.info(f"RNN Test Loss: {loss}")
