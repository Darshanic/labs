# Import libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams

# Corpus
corpus = [
    "neural networks are powerful",
    "word embedding captures meaning",
    "deep learning models learn representation",
    "natural language processing uses embedding"
]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(corpus)

# Generate skip-gram pairs
pairs = []
labels = []

window_size = 2

for seq in sequences:
    sg = skipgrams(seq, vocabulary_size=vocab_size, window_size=window_size)
    pairs.extend(sg[0])
    labels.extend(sg[1])

pairs = np.array(pairs)
labels = np.array(labels)

# Split target and context
target_words = pairs[:, 0]
context_words = pairs[:, 1]

# Build model
embedding_dim = 50

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1, name='embedding'),
    tf.keras.layers.Reshape((embedding_dim,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(target_words, labels, epochs=50, verbose=1)

# Get embeddings
embedding_layer = model.get_layer('embedding')
embedding_weights = embedding_layer.get_weights()[0]

# Example: print embedding of a word
word = "neural"
print(f"Embedding for '{word}':")
print(embedding_weights[word_index[word]])