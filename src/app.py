from flask import Flask, send_from_directory, request, flash, jsonify
from flask_cors import CORS

from models.generate_text import TextGenerator
from models.subprompt_generator import SubPromptgenerator
from models.LSTM.generate import LSTMTextGenerator

app = Flask(__name__)
CORS(app)


generator = TextGenerator("./src/models/gpt-neo-125M")
subpromptGenerator = SubPromptgenerator()
engLstmGenerator = LSTMTextGenerator(model_path="./src/models/LSTM/model_weights_english_v1.hdf5",
                                     tokenizer_path='./src/models/LSTM/tokenizer_english.pickle', rwm_path='./src/models/LSTM/reverse_word_map_english.json')
assameseLstmGenerator = LSTMTextGenerator(model_path="./src/models/LSTM/model_weights_assamese_v1.hdf5",
                                          tokenizer_path='./src/models/LSTM/tokenizer_assamese.pickle', rwm_path='./src/models/LSTM/reverse_word_map_assamese.json')


@app.route("/test", methods=["GET"])
def test():
    return "<h1>Working...</h1>"


@app.route("/get-outline", methods=['POST'])
def generateOutline():
    data = request.json
    prompt = data['prompt']
    msg = ''

    if not prompt:
        msg = 'Text Prompt is required'
    else:
        generated_sbp = subpromptGenerator.generate(prompt)
        msg = 'successfully generated'

    return jsonify({'outline': generated_sbp[:5], 'message': msg})


@app.route("/generate-text", methods=["POST"])
def generateText():
    data = request.json
    prompt = data['prompt']
    sub_prompts = data['subprompts']
    msg = ''

    if not prompt:
        msg = 'Text Prompt is required'
    elif not sub_prompts:
        msg = 'Sub Prompts is required'
    else:
        intro, paras = generator.generateBlog(prompt, sub_prompts)
        msg = 'successfully generated'
        generator.clean()

    return jsonify({'intro': intro, 'paragraphs': paras, 'message': msg})


@app.route("/get-image", methods=["POST"])
def generateImage():
    prompt = request.form['prompt']
    numberOfImages = request.form['numberOfImages']
    msg = ''

    if not prompt:
        msg = 'Text Prompt is required'
    elif not numberOfImages:
        msg = 'Number of words is required'
    else:
        msg = 'successfully generated'

    return jsonify({'message': msg})


@app.route("/generate-post-lstm", methods=['POST'])
def generatePost():
    data = request.json
    prompt = data['prompt']
    num_of_words = data['numberOfWords']
    num_of_words = int(num_of_words)
    language = data['language']  # english, assamese
    msg = ''

    if not prompt:
        msg = "Text prompt is required"
    elif not num_of_words:
        msg = "Number of words is required"
    else:
        if language == "assamese":
            generated_text = assameseLstmGenerator.generateText(
                prompt, num_of_words)
        else:
            generated_text = engLstmGenerator.generateText(
                prompt, num_of_words)
        msg = 'successfully generated'

    return jsonify({'message': msg, 'text': generated_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4040, debug=True)
