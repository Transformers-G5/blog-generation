from flask import Flask, send_from_directory, request, flash, jsonify
from flask_cors import CORS

from models.generate_text import TextGenerator
from models.subprompt_generator import SubPromptgenerator

app = Flask(__name__)
CORS(app)


generator = TextGenerator("./src/models/gpt-neo-125M")
subpromptGenerator = SubPromptgenerator()


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

    return jsonify({'outline': generated_sbp, 'message': msg})


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


@app.route("/generate-text", methods=["POST"])
def generateText():
    data = request.json
    prompt = data['prompt']
    sub_prompts = data['subprompts']
    msg = ''
    generated_text = ''
    numberOfWords = 300

    if not prompt:
        msg = 'Text Prompt is required'
    elif not sub_prompts:
        msg = 'Sub Prompts is required'
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
