from config import EMOJIS, DATE, NO_BPE_ITERATIONS, RANDOM_STATE, EMBEDDING_DIM, EMBEDDING_MODEL_EPOCHS
import re
from collections import defaultdict
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import logging
from gensim.models import KeyedVectors


class BytePairEncoding:
    def __init__(self, no_iterations=None, corpora=None):
        self.no_iterations = no_iterations
        self.corpora = corpora

    def _get_stats(self, vocab):
        pairs = defaultdict(int)
        
        for word, freq in vocab.items():
            symbols = word.split()
            
            for index in range(len(symbols)-1):
                pairs[symbols[index],symbols[index+1]] += freq
        
        return pairs

    def _merge_vocab(self, pair, vocab_i):
        vocab_ii = {}
        pair_str = ' '.join(pair)
        joined_pair = ''.join(pair)
        
        for word in vocab_i:
            word_i = word.replace(pair_str, joined_pair)
            vocab_ii[word_i] = vocab_i[word]

        return vocab_ii

    def _get_vocab(self, data):
        vocab = defaultdict(int)
        
        for line in data:
            for word in str(line).split():
                vocab[' '.join(list(word)) + ' </w>'] += 1
        
        return vocab

    def learn(self):
        data = self.corpora
        n = self.no_iterations
        corpus = ' '.join(data)
        vocab_set = list(set([letter.replace(' ', '</w>') for letter in corpus.replace('\n', '')]))
        vocab_set_i = list(set([letter.replace(' ', '</w>') for letter in corpus.replace('\n', '')]))
        vocab = self._get_vocab(data)
        
        for _ in tqdm(range(n), desc="BPE iterations"):
            pairs = self._get_stats(vocab)
            best = max(pairs, key=pairs.get)
            vocab_set.append(best)
            vocab_set_i.append(''.join(best))
            vocab = self._merge_vocab(best, vocab)

        self.vocab_set = vocab_set
        self.vocabulary = vocab_set_i

    def tokenize(self, corpus):
        vocab_set = self.vocab_set
        bests = vocab_set[::-1]
        vocab = self._get_vocab([corpus])
        
        while True:
            if len(bests) == 0:
                break

            best = bests.pop()
            vocab = self._merge_vocab(best, vocab)

        token_dict = vocab
        tokens = list(token_dict.keys())
        list_of_tokens = []

        for word in [token.split() for token in tokens]:
            list_of_tokens += word

        return list_of_tokens
    
    def save(self, file_path):
        model_data = {
            'vocab_set': self.vocab_set,
            'vocabulary': self.vocabulary,
        }

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        self.vocab_set = model_data['vocab_set']
        self.vocabulary = model_data['vocabulary']



    def get_one_hot(self, token):
        vocab = {token: idx for idx, token in enumerate(self.vocabulary)}
        one_hot_matrix = np.eye(len(self.vocabulary))
        index = vocab[token]

        return one_hot_matrix[index]


class FastText:
    def __init__(self, embedding_dim=100, learning_rate=0.1, epochs=5, context_window=5):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.context_window = context_window
        self.vocab = defaultdict(int)
        self.word_to_index = {}
        self.index_to_word = {}
        self.embeddings = None
        self.context_embeddings = None

    def save_model(self, file_path):
        """Save the model to a file."""
        model_data = {
            'embedding_dim': self.embedding_dim,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'context_window': self.context_window,
            'vocab': self.vocab,
            'word_to_index': self.word_to_index,
            'index_to_word': self.index_to_word,
            'embeddings': self.embeddings,
            'context_embeddings': self.context_embeddings,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        logging.info(f"Model saved to {file_path}")

    @classmethod
    def load_model(cls, file_path):
        """Load the model from a file."""
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        model = cls(
            embedding_dim=model_data['embedding_dim'],
            learning_rate=model_data['learning_rate'],
            epochs=model_data['epochs'],
            context_window=model_data['context_window']
        )
        model.vocab = model_data['vocab']
        model.word_to_index = model_data['word_to_index']
        model.index_to_word = model_data['index_to_word']
        model.embeddings = model_data['embeddings']
        model.context_embeddings = model_data['context_embeddings']
        logging.info(f"Model loaded from {file_path}")
        return model

    def build_vocab(self, corpus):
        for sentence in corpus:
            for word in sentence:
                self.vocab[word] += 1
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab.keys())}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        vocab_size = len(self.vocab)
        self.embeddings = np.random.uniform(-1, 1, (vocab_size, self.embedding_dim))
        self.context_embeddings = np.random.uniform(-1, 1, (vocab_size, self.embedding_dim))
        
    def train(self, corpus):
        epoch_losses = []  
        for epoch in range(self.epochs):
            total_loss = 0
            for sentence in tqdm(corpus, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                sentence_indices = [self.word_to_index[word] for word in sentence if word in self.word_to_index]
                for i, target_idx in enumerate(sentence_indices):
                    start = max(i - self.context_window, 0)
                    end = min(i + self.context_window + 1, len(sentence_indices))
                    context_indices = sentence_indices[start:i] + sentence_indices[i + 1:end]
                    for context_idx in context_indices:
                        total_loss += self._train_pair(target_idx, context_idx)
            epoch_losses.append(total_loss)  
            logging.info(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
        
        logging.info(f"Plotting the loss ...")
        logging.info(f"Losses:{range(1, self.epochs + 1), epoch_losses}")

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, self.epochs + 1), epoch_losses, marker='o', label='Loss')
        plt.title('Epoch vs Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("plots", f"epoch_loss_plot_fasttext {DATE}.png"))

    def _train_pair(self, target_idx, context_idx):
        target_embedding = self.embeddings[target_idx]
        context_embedding = self.context_embeddings[context_idx]
        score = np.dot(target_embedding, context_embedding)
        prob = self._sigmoid(score)

        error = 1 - prob  
        d_target = error * context_embedding
        d_context = error * target_embedding

        self.embeddings[target_idx] += self.learning_rate * d_target
        self.context_embeddings[context_idx] += self.learning_rate * d_context

        return -np.log(prob)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_embedding(self, word):
        if word in self.word_to_index:
            return self.embeddings[self.word_to_index[word]]
        else:
            raise ValueError(f"Word '{word}' not in vocabulary.")
        
    def get_embeddings(self, sentence):
        return [self.get_embedding(word) for word in sentence if word in self.word_to_index]
    
    def get_average_embedding(self, sentence):
        embeddings = self.get_embeddings(sentence)

        return np.mean(embeddings, axis=0)
    

class GoogleNewsEmbedding:
    def __init__(self, model_path):
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def get_embedding(self, word):
        return self.model[word]

    def get_average_embedding(self, sentence):
        embeddings = [self.get_embedding(word) for word in sentence if word in self.model]
        return np.mean(embeddings, axis=0)


if __name__ == "__main__":
    use_existing_bpe_model = True

    if use_existing_bpe_model:
        corpora = pd.read_csv(os.path.join("corpora", "Reviews.csv")).sample(n=10000, random_state=RANDOM_STATE)["Text"]
        new_BPE = BytePairEncoding(no_iterations=NO_BPE_ITERATIONS, corpora=None)
        new_BPE.load(os.path.join("models", "bpe-2048"))

    else:
        # Load Larger Corpora
        corpora = pd.read_csv(os.path.join("corpora", "Reviews.csv")).sample(n=10000, random_state=RANDOM_STATE)["Text"]

        # # Add Emoji Row
        # emoji_row = " ".join(EMOJIS)
        # corpora = pd.concat([corpora, pd.Series([emoji_row])], ignore_index=True) # Uncomment this line to add the emoji row

        # Learn Tokens
        BPE = BytePairEncoding(NO_BPE_ITERATIONS, corpora)
        BPE.learn()

        BPE.save(os.path.join("models", f"bpe {DATE}"))

        new_BPE = BytePairEncoding(NO_BPE_ITERATIONS, corpora)
        new_BPE.load(os.path.join("models", f"bpe {DATE}"))

    # One-hot vector for the token 't'
    one_hot_vector = new_BPE.get_one_hot('t')

    logging.info(f"One-hot vector for the token 't': {one_hot_vector}")
    logging.info(f"Vocabulary: {new_BPE.vocabulary}")
    logging.info(f"Vocabulary Set: {new_BPE.vocab_set}")

    # Tokenize Corpus
    tokenized_series = corpora.progress_apply(new_BPE.tokenize)

    logging.info(f"Tokenized Series: {corpora.head(), tokenized_series.head()}")

    logging.info("Training FastText Model ...")

    # Train Embedding
    use_existing_fasttext_model = True

    if use_existing_fasttext_model:
        new_fasttext_model = FastText(embedding_dim=EMBEDDING_DIM, epochs=EMBEDDING_MODEL_EPOCHS)
        new_fasttext_model.load_model(os.path.join("models", "fasttext-300-40"))
        
    else:
        fasttext_model = FastText(embedding_dim=EMBEDDING_DIM, epochs=EMBEDDING_MODEL_EPOCHS)
        fasttext_model.build_vocab(tokenized_series)
        fasttext_model.train(tokenized_series)

        # Save the model
        fasttext_model.save_model(os.path.join("models", f"fasttext {DATE}"))

        logging.info("Loading the model ...")

        # Load the model
        new_fasttext_model = fasttext_model.load_model(os.path.join("models", f"fasttext {DATE}"))

    # Get the embedding for a specific word
    word = new_BPE.tokenize("Bilkent University is a private university located in Ankara, Turkey!")
    avg_embedding = new_fasttext_model.get_average_embedding(word)
    logging.info(f"Average Embedding for '{word}': {avg_embedding}")

    # Load Google News Word2Vec model for comparison
    google_news_model_path = os.path.join("models", "GoogleNews-vectors-negative300.bin.gz")
    google_news_embedding = GoogleNewsEmbedding(google_news_model_path)

    sentence = "Bilkent University is a private university located in Ankara, Turkey!"
    tokenized_sentence = new_BPE.tokenize(sentence)
    google_news_avg_embedding = google_news_embedding.get_average_embedding(tokenized_sentence)
    logging.info(f"Google News Average Embedding for '{sentence}': {google_news_avg_embedding}")
