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



# Transformers
from utils.fromaters.letter_formater import simple_letter_formater, createLoveLetterPlaceHolder
from utils.postprocessors.common_postprocessors import remove_start_end, capitalize_first, decorate_response
from utils.preprocessors.common_preprocessor import remove_tags

from models.transformers.model import Main

blogMain = Main()
textGeneratorLanguageModelBlog = blogMain.build(
    model_path='src/models/transformers/blogs_v8.1_loss_0-2.06_model', vocab_path='src/models/transformers/blogs_vocab_v8.1_0.06')

restaurantMain = Main()
textGeneratorLanguageModelRestaurant = restaurantMain.build(model_path='src/models/transformers/marketing_restaurant_loss_0.08_model',
                                          vocab_path='src/models/transformers/marketing_restaurant_vocab_loss_0.08')

schoolMain = Main()
textGeneratorLanguageModelSchool = schoolMain.build(
    model_path='src/models/transformers/marketing_school_loss_0.066_model', vocab_path='src/models/transformers/marketing_school_vocab_loss_0.06')

# main4 = Main()
# textGeneratorLanguageModel4 = main4.build(
#     model_path='src/models/transformers/chat_loss_0.11_model', vocab_path='src/models/transformers/chat_vocab_loss_0.11')

loveLetterMain = Main()
textGeneratorLanguageModelLoveLetter = loveLetterMain.build(
    model_path='src/models/transformers/letter_loss_0 (1).03_model', vocab_path='src/models/transformers/letter_vocab_loss_0 (1).03')

emailMain = Main(n_layer=12)
textGeneratorLanguageModelEmail = emailMain.build(
    model_path='src/models/transformers/email_marketing_loss_0.11_model', vocab_path='src/models/transformers/email_marketing_vocab_loss_0.11')




@app.route('/api/gentext/blog', methods=['POST'])
def blogAPI():
    data = request.get_json()

    prompt = data.get('prompt')
    max_len = data.get('max_len')
    result = textGeneratorLanguageModelBlog.generate(
        inputs={"prompt": prompt}, max_len=max_len)

    response = {
        'gen_text': result
    }

    return jsonify(response)


@app.route('/api/gentext/marketing/restaurant', methods=['POST'])
def marketingResturantAPI():
    data = request.get_json()

    prompt = data.get('prompt')
    max_len = data.get('max_len')
    name = data.get('name')

    if prompt == '':
        prompt = '[start]'

    result = textGeneratorLanguageModelRestaurant.generate(
        inputs={"prompt": prompt}, max_len=max_len)
    result = result.split('[end]')[0].replace(
        '[start]', '')  # remove the [star] and [end] tokens
    if len(name.strip()) != 0:
        result = result.replace('[restaurant name]', f'<b>{name}</b>')

    # capitalize First later of every sentence
    result = result.split(".")[:-1]
    result = [t.lstrip() for t in result]
    result = [t[0].upper() + t[1:] + '. ' for t in result]
    result = "".join(result)

    response = {
        'gen_text': result
    }

    return jsonify(response)


@app.route('/api/gentext/marketing/school', methods=['POST'])
def marketingSchoolAPI():
    data = request.get_json()

    prompt = data.get('prompt')
    max_len = data.get('max_len')
    name = data.get('name')

    if prompt == '':
        prompt = '[start]'

    result = textGeneratorLanguageModelSchool.generate(
        inputs={"prompt": prompt}, max_len=max_len)
    result = result.split('[end]')[0].replace(
        '[start]', '')  # remove the [star] and [end] tokens

    if len(name.strip()) != 0:
        result = result.replace('[school name]', f'<b>{name}</b>')

    # capitalize First later of every sentence
    result = result.split(".")[:-1]
    result = [t.lstrip() for t in result]
    result = [t[0].upper() + t[1:] + '. ' for t in result]
    result = "".join(result)

    response = {
        'gen_text': result
    }

    return jsonify(response)

@app.route('/api/gentext/letter/love', methods=['POST'])
def loveLetterAPI():
    data = request.get_json()

    prompt = data.get('prompt')
    max_len = data.get('max_len')
    to_name = data.get('to_name') or None
    from_name = data.get('from_name') or None

    if prompt == '':
        prompt = '[start]'

    prompt = remove_tags(prompt)
    result = textGeneratorLanguageModelLoveLetter.generate(
        inputs={"prompt": prompt}, max_len=max_len, stop_at='[end]')
    result = remove_start_end(result)

    loveLetterPlaceHolder = createLoveLetterPlaceHolder(
        to_name=to_name, from_name=from_name)
    result = simple_letter_formater(result, placeholders=loveLetterPlaceHolder)

    response = {
        'gen_text': result
    }

    return jsonify(response)


@app.route('/api/gentext/marketing/email', methods=['POST'])
def emailMarketingAPI():
    data = request.get_json()

    prompt = data.get('prompt')
    max_len = data.get('max_len')
    to_name = data.get('to_name') or None
    from_name = data.get('from_name') or None

    if prompt == '':
        prompt = '[start]'

    prompt = remove_tags(prompt)
    result = textGeneratorLanguageModelEmail.generate(
        inputs={"prompt": prompt}, max_len=max_len, stop_at='[end]')

    result = remove_start_end(result)

    result = simple_letter_formater(result)

    response = {
        'gen_text': result
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4040, debug=True)

