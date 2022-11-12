from flask import Flask, send_from_directory, request, flash, jsonify
from flask_cors import CORS

from models.generate_text import TextGenerator

app = Flask(__name__)
CORS(app)


@app.route("/test", methods=["GET"])
def test():
    return "<h1>Working...</h1>"


@app.route("/generate-text", methods=["POST"])
def generateText():
    data = request.json
    print(data['prompt'])
    prompt = data['prompt']
    numberOfWords = data['numberOfWords']
    msg = ''
    generated_text = ''

    if not prompt:
        msg = 'Text Prompt is required'
    elif not numberOfWords:
        msg = 'Number of words is required'
    else:
        generator = TextGenerator("./src/models/gpt-neo-125M")
        generated_text = generator.generateText(prompt, numberOfWords)
        msg = 'successfully generated'

    return jsonify({'text': generated_text, 'message': msg})


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4040, debug=True)
