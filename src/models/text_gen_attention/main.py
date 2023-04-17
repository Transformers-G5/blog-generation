import numpy as np

import typing
from typing import Any, Tuple

import einops
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

import tensorflow as tf
import tensorflow_text as tf_text
import pathlib
import pickle



@tf.keras.utils.register_keras_serializable(package='Custom', name=None)
def tf_lower_and_split_punct(text):
    # Split accented characters
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # add space arround punctuation
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # remove non-desplayable characters
    text = tf.strings.regex_replace(text, '[^\x00-\x7F]+', '')
    #Strip white space
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

'''
The encoder:
  1. Takes a list of token IDs (from context_text_processor).
  2. Looks up an embedding vector for each token (Using a layers.Embedding).
  3. Processes the embeddings into a new sequence (Using a bidirectional layers.GRU).
  4. Returns the processed sequence. This will be passed to the attention head.
'''


class Encoder(tf.keras.layers.Layer):
  def __init__(self, text_processor, units):
    super(Encoder, self).__init__()
    self.text_processor = text_processor
    self.vocab_size = text_processor.vocabulary_size()
    self.units = units

    #The embedding layer converts tokens into vectors
    self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.units, mask_zero=True)

    #The RNN layer processes those vectors sequentially
    self.rnn = tf.keras.layers.Bidirectional(merge_mode='sum', 
                                             layer=tf.keras.layers.GRU(self.units, return_sequences=True, recurrent_initializer='glorot_uniform' ))
  
  def call(self, x):
    x = self.embedding(x)
    x = self.rnn(x)
    # print('call')
    return x

  def convert_input(self, texts):
    texts = tf.convert_to_tensor(texts)
    if len(texts.shape) == 0:
      texts = tf.convert_to_tensor(texts)[tf.newaxis]
    context = self.text_processor(texts).to_tensor()
    context = self(context)
    return context

class CrossAttention(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__()

    self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x, context):
    atten_output, atten_score = self.mha(query=x, value=context, return_attention_scores=True)
    x = self.add([x, atten_output])
    x = self.layernorm(x)
    return x
'''
The decoder's job is to generate predictions for the next token at each location in the target sequence.
  1. It looks up embeddings for each token in the target sequence.
  2. It uses an RNN to process the target sequence, and keep track of what it has generated so far.
  3. It uses RNN output as the "query" to the attention layer, when attending to the encoder's output.
  4. At each location in the output it predicts the next token.
'''

class Decoder(tf.keras.layers.Layer):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, text_processor, units):
    super(Decoder, self).__init__()
    self.text_processor = text_processor
    self.vocab_size = text_processor.vocabulary_size()
    self.word_to_id = tf.keras.layers.StringLookup(vocabulary=text_processor.get_vocabulary(), mask_token='', oov_token='[UNK]')
    self.id_to_word = tf.keras.layers.StringLookup(vocabulary=text_processor.get_vocabulary(), mask_token='', oov_token='[UNK]', invert=True)
    self.start_token = self.word_to_id('[START]')
    self.end_token = self.word_to_id('[END]')

    self.units = units

    # 1. The embedding layer converts token IDs to vectors
    self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.units, mask_zero=True)
    # 2. The RNN keeps track of what's been generated so far.
    self.rnn = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    #3. The RNN output will be the query for the attention layer.
    self.attention = CrossAttention(self.units)

    # self.fc1 = tf.keras.layers.Dense(self.units, activation='relu')
    # 4. This fully connected layer produces the logits for each output token.
    self.output_layer = tf.keras.layers.Dense(self.vocab_size)

  def call(self, context, x, state=None, return_state=False):
    #Lookup for embeddings
    x = self.embedding(x)
    #Process the target sequence
    x, state = self.rnn(x, initial_state=state)
    #Use the rnn output as the query for the attention over the context
    x = self.attention(x, context)

    # x = self.fc1(x)

    #generate logit predictyions for the next token
    logits = self.output_layer(x)


    if return_state:
      return logits, state
    return logits


@Decoder.add_method
def get_initial_state(self, context):
  batch_size = tf.shape(context)[0]
  print(batch_size)
  start_tokens = tf.fill([batch_size, 1], self.start_token)
  done = tf.zeros([batch_size, 1], dtype=tf.bool)
  embedded = self.embedding(start_tokens)
  return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

@Decoder.add_method
def tokens_to_text(self, tokens):
  words = self.id_to_word(tokens)
  result = tf.strings.reduce_join(words, axis=-1, separator=' ')
  result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
  result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
  return result

@Decoder.add_method
def get_next_token(self, context, next_token, done, state, temperature = 0.0):

  logits, state = self(context, next_token, state = state, return_state=True) 

  if temperature == 0.0:
    next_token = tf.argmax(logits, axis=-1)
  else:
    logits = logits[:, -1, :]/temperature
    next_token = tf.random.categorical(logits, num_samples=1)

  # If a sequence produces an `end_token`, set it `done`
  # done = done | (next_token == self.end_token)
  # # Once a sequence is done it only produces 0-padding.
  # next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

  return next_token, done, state


class TextGen(tf.keras.Model):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, units, context_text_processor, target_text_processor):
    super().__init__()
    #Build the encoder and the decoder
    self.encoder = Encoder(context_text_processor, units)
    self.decoder = Decoder(context_text_processor, units)

  def call(self, inputs):
    # print("here1")
    context, x = inputs
    # print("here2")

    context = self.encoder(context)
    # print("here3")

    logits = self.decoder(context, x)
    # print("here4")

    try:
      del logits._keras_mask
    except AttributeError:
      pass
    # print("here5", logits)
    return logits
  
  def gen(self,
                texts, *,
                max_length=50,
                temperature=0.0):
    # Process the input texts
    context = self.encoder.convert_input(texts)
    batch_size = tf.shape(texts)[0]

    # Setup the loop inputs
    tokens = []
    attention_weights = []
    next_token, done, state = self.decoder.get_initial_state(context)

    for _ in range(max_length):
      # Generate the next token
      next_token, done, state = self.decoder.get_next_token(context, next_token, done,  state, temperature)

      # Collect the generated tokens
      tokens.append(next_token)
      # attention_weights.append(self.decoder.last_attention_weights)

      # if tf.executing_eagerly() and tf.reduce_all(done):
      #   break

    # Stack the lists of tokens and attention weights.
    tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)
    # self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)

    result = self.decoder.tokens_to_text(tokens)
    return result



class Generator():
    def __init__(self, model_path, path_to_vectorizer_config, path_to_vectorizer_weights):
       
        self.model = self._create_model(model_path, path_to_vectorizer_config, path_to_vectorizer_weights)


    def _create_model(self, path_to_model, path_to_vectorizer_config, path_to_vectorizer_weights, units=256):
        loaded_config = pickle.load(open(path_to_vectorizer_config, 'rb'))
        loaded_weights = pickle.load(open(path_to_vectorizer_weights, 'rb'))

        text_processor = tf.keras.layers.TextVectorization.from_config(loaded_config)
        text_processor.set_weights(loaded_weights)

        new_model = TextGen(units, text_processor, text_processor)
        new_model.load_weights(path_to_model)

        # print(new_model.gen(["hello"], 50))
        return new_model
    
    def generate(self, inputs, max_word=50):
        prompt = inputs['prompt']
        result = self.model.gen(texts=[prompt], max_length=max_word)
        result = result[0].numpy().decode()
        return result

if __name__ == '__main__':
    textgen = Generator(model_path='../../../notebooks/40_model.tf', 
                path_to_vectorizer_config='../../../notebooks/text_processor_config.pkl', path_to_vectorizer_weights='../../../notebooks/text_processor_weights.pkl')
    
    
    result = textgen.generate(inputs={'prompt':"Life"}, max_word=100)
    print()
    print()
    print("RESULT: ", result)