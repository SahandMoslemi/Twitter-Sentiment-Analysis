import re
from collections import defaultdict
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle


class BytePairEncoding:
    def __init__(self, no_iterations, corpora):
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
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in vocab_i:
            word_i = p.sub(''.join(pair), word)
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
    
    def get_one_hot(self, token):
        vocab = {token: idx for idx, token in enumerate(self.vocabulary)}
        one_hot_matrix = np.eye(len(self.vocabulary))
        index = vocab[token]

        return one_hot_matrix[index]


class FastText:
    def __init__(self, embedding_dim=100, learning_rate=0.01, epochs=5, context_window=5):
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
        print(f"Model saved to {file_path}")

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
        print(f"Model loaded from {file_path}")
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
        for epoch in range(self.epochs):
            total_loss = 0
            for sentence in tqdm(corpus, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                sentence_indices = [self.word_to_index[word] for word in sentence if word in self.word_to_index]
                for i, target_idx in enumerate(sentence_indices):
                    # Context window
                    start = max(i - self.context_window, 0)
                    end = min(i + self.context_window + 1, len(sentence_indices))
                    context_indices = sentence_indices[start:i] + sentence_indices[i + 1:end]
                    for context_idx in context_indices:
                        total_loss += self._train_pair(target_idx, context_idx)
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    def _train_pair(self, target_idx, context_idx):
        # Forward pass
        target_embedding = self.embeddings[target_idx]
        context_embedding = self.context_embeddings[context_idx]
        score = np.dot(target_embedding, context_embedding)
        prob = self._sigmoid(score)

        # Loss and gradients
        error = 1 - prob  # Skip-gram with negative sampling simplified
        d_target = error * context_embedding
        d_context = error * target_embedding

        # Update embeddings
        self.embeddings[target_idx] += self.learning_rate * d_target
        self.context_embeddings[context_idx] += self.learning_rate * d_context

        # Log loss
        return -np.log(prob)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_embedding(self, word):
        if word in self.word_to_index:
            return self.embeddings[self.word_to_index[word]]
        else:
            raise ValueError(f"Word '{word}' not in vocabulary.")
        
    def get_average_embedding(self, sentence):
        embeddings = [self.get_embedding(word) for word in sentence if word in self.word_to_index]

        return np.mean(embeddings, axis=0)


if __name__ == '__main__':
    # Corpora
    # Parameters
    corpora = pd.read_csv(os.path.join('corpora', 'Reviews.csv')).sample(n=10000, random_state=1)['Text']
    no_bpe_iterations = 32

    # Learn Tokens
    BPE = BytePairEncoding(no_bpe_iterations, corpora)
    BPE.learn()

    # # One-hot vector for the token 't'
    # one_hot_vector = BPE.get_one_hot('t')

    # Target Data
    training_df = (
    pd.read_csv(os.path.join('data', 'twitter_training.csv')).rename(columns={'Positive': 'sentiment', 'im getting on borderlands and i will murder you all ,': 'text'}).sample(n=5000, random_state=2))

    tokenized_series = training_df["text"].apply(BPE.tokenize)

    # Train Embedding
    fasttext_model = FastText(embedding_dim=128, epochs=100)
    fasttext_model.build_vocab(tokenized_series)
    fasttext_model.train(tokenized_series)

    fasttext_model.save_model("fasttext_1")
    new_fasttext_model = fasttext_model.load_model("fasttext_1")

    # Get the embedding for a specific word
    word = BPE.tokenize("Bilkent!")
    embedding = fasttext_model.get_average_embedding(word)
    print(f"Embedding for '{word}': {embedding}")


