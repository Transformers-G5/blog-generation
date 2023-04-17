import numpy as np
import pickle
import json
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences


class LSTMTextGenerator:
    def __init__(self, model_path, tokenizer_path, rwm_path) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.rwm_path = rwm_path
        self.tokenizer = None
        self.reverse_word_map = None
        self.model = None
        self._load_tokenizer_and_reverse_word_map()
        self._load_model()

    def _load_tokenizer_and_reverse_word_map(self):
        with open(self.tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open(self.rwm_path) as json_file:
            self.reverse_word_map = json.load(json_file)

    def _load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)
        print(self.model_path, " : ", " >> model loaded.")

    def generateText(self, prompt="", max_length=20):
        ''' Generates a sequence given a string seq using specified model until the total sequence length
        reaches max_len'''

        # Tokenize the input string
        tokenized_sent = self.tokenizer.texts_to_sequences([prompt])
        max_length = max_length + len(tokenized_sent[0])
        # If sentence is not as long as the desired sentence length, we need to 'pad sequence' so that
        # the array input shape is correct going into our LSTM. the `pad_sequences` function adds
        # zeroes to the left side of our sequence until it becomes 19 long, the number of input features.
        while len(tokenized_sent[0]) < max_length:
            padded_sentence = pad_sequences(tokenized_sent[-19:], maxlen=19)
            op = self.model.predict(np.asarray(padded_sentence).reshape(1, -1))
            tokenized_sent[0].append(op.argmax()+1)

        return " ".join(map(lambda x: self.reverse_word_map[str(x)], tokenized_sent[0]))


if __name__ == "__main__":
    # to generate assamese text
    # lstmTextGen = LSTMTextGenerator(model_path="./src/models/LSTM/model_weights_assamese_v1.hdf5",
    #                                 tokenizer_path='./src/models/LSTM/tokenizer_assamese.pickle', rwm_path='./src/models/LSTM/reverse_word_map_assamese.json')
    # prompt = "এক ম'ল পানীৰ ভৰ হ'ব প্ৰায়"
    # res = lstmTextGen.generateText(prompt)
    # print(res)

    # to generate english text
    engTextGen = LSTMTextGenerator(model_path="./src/models/LSTM/model_weights_english_v1.hdf5",
                                   tokenizer_path='./src/models/LSTM/tokenizer_english.pickle', rwm_path='./src/models/LSTM/reverse_word_map_english.json')
    prompt = "Nice day at the beach"
    res = engTextGen.generateText(prompt)
    print(res)
