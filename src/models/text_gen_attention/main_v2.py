import tensorflow as tf
import tensorflow_text as tf_text
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class Generator:
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)
    def generate(self, inputs, verbose=False):
        if verbose:
            print("Generating..")
        prompt = inputs['prompt']
        result = self.model.generate(tf.constant([prompt]))
        result = result[0].numpy().decode()
        return result
    def discardAfterEnd(self, string):
        return string.split("[END]")[0]

gen = Generator(model_path='../../../models/attention/4_model_')

test_inputs = ['Negative Emotions', "Anger issues"]


for _input in test_inputs:
    res = gen.discardAfterEnd(gen.generate(inputs={'prompt':_input}))
    print()
    print()
    print(_input, res)