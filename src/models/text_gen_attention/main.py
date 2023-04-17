import tensorflow as tf
import tensorflow_text as tf_text

class Generator():
    def __init__(self, model_path):
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

        self.loaded_model = tf.keras.models.load_model(model_path, custom_objects={'tf_lower_and_split_punct': tf_lower_and_split_punct}, compile=False)
    
    def generate(self, inputs, max_word=50):
        prompt = inputs['prompt']
        result = self.loaded_model.gen([prompt], max_word)
        result = result[0].numpy().decode()
        return result

if __name__ == '__main__':
    textgen = Generator(model_path='../../../notebooks/5_model.tf')
    textgen.generate(inputs={'prompt':"Hello World"})