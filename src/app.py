from flask import Flask, send_from_directory, request, flash, jsonify
from flask_cors import CORS

from models.generate_text import TextGenerator
# from models.subprompt_generator import SubPromptgenerator
from models.LSTM.generate import LSTMTextGenerator

app = Flask(__name__)
CORS(app)


generator = TextGenerator("./src/models/gpt-neo-125M")
# subpromptGenerator = SubPromptgenerator()
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
# from utils.fromaters.letter_formater import simple_letter_formater, createLoveLetterPlaceHolder
# from utils.postprocessors.common_postprocessors import remove_start_end, capitalize_first, decorate_response
# from utils.preprocessors.common_preprocessor import remove_tags

# from models.transformers.model import Main

# blogMain = Main()
# textGeneratorLanguageModelBlog = blogMain.build(
#     model_path='src/models/transformers/blogs_v8.1_loss_0-2.06_model', vocab_path='src/models/transformers/blogs_vocab_v8.1_0.06')

# restaurantMain = Main()
# textGeneratorLanguageModelRestaurant = restaurantMain.build(model_path='src/models/transformers/marketing_restaurant_loss_0.08_model',
#                                           vocab_path='src/models/transformers/marketing_restaurant_vocab_loss_0.08')

# schoolMain = Main()
# textGeneratorLanguageModelSchool = schoolMain.build(
#     model_path='src/models/transformers/marketing_school_loss_0.066_model', vocab_path='src/models/transformers/marketing_school_vocab_loss_0.06')

# # main4 = Main()
# # textGeneratorLanguageModel4 = main4.build(
# #     model_path='src/models/transformers/chat_loss_0.11_model', vocab_path='src/models/transformers/chat_vocab_loss_0.11')

# loveLetterMain = Main()
# textGeneratorLanguageModelLoveLetter = loveLetterMain.build(
#     model_path='src/models/transformers/letter_loss_0 (1).03_model', vocab_path='src/models/transformers/letter_vocab_loss_0 (1).03')

# emailMain = Main(n_layer=12)
# textGeneratorLanguageModelEmail = emailMain.build(
#     model_path='src/models/transformers/email_marketing_loss_0.11_model', vocab_path='src/models/transformers/email_marketing_vocab_loss_0.11')





# from model.bytelevel.model_v2 import setup as setupv2

from models.transformers.bytelevel.model_v3 import setup

vocab_pathv2 = 'src/models/transformers/bytelevel/models/vocab_blog/chetna-md-16k-5f-vocab.json'
merges_pathv2 = 'src/models/transformers/bytelevel/models/vocab_blog/chetna-md-16k-5f-merges.txt'
blog_model_path = 'src/models/transformers/bytelevel/models/chetna_sm_12layer_12head_128block_1687531783.6161177_finetune_gptlike_model'
TextGeneratorBlog = setup(vocab_path=vocab_pathv2, merges_path=merges_pathv2,
                        block_size=128, num_layer=12, num_head=12, num_embd=768)
blogGenerator = TextGeneratorBlog(
    model_path=blog_model_path)



@app.route('/api/gentext/blog', methods=['POST'])
def blogAPI():
    data = request.get_json()

    prompt = data.get('prompt')
    max_len = data.get('max_len')
    temperature = data.get('temperature')
    print(prompt)
    print("temp", temperature)
    if temperature is None: temperature = 1.0

    result, result_topk = blogGenerator.generate(
        inputs={"prompt": prompt}, stop_token_id=[blogGenerator.end_token_id], max_len=max_len, verbose=True, temperature=temperature or 1.0)
    prompt_len = len(blogGenerator.tokenizer.decode(
        blogGenerator.tokenizer.encode(prompt).ids))

    result = result.strip()
    result = result[prompt_len:]
    result_topk = result_topk[prompt_len:]

    response = {
        'text': result,
        'more': [{'text': result_topk, 'info':f'Top k sampling, k=1, temp={temperature}'}]
    }

    return jsonify(response)


vocab_pathv3 = 'src/models/transformers/bytelevel/models/vocab/chetna-md-16k-5f-vocab.json'
merges_pathv3 = 'src/models/transformers/bytelevel/models/vocab/chetna-md-16k-5f-merges.txt'
email_model_path = 'src/models/transformers/bytelevel/models/chetna_sm_12layer_12head_128block_1687630399.0975342_finetune_gptlike_newvocab_email_model'
TextGeneratorEmail = setup(vocab_path=vocab_pathv3, merges_path=merges_pathv3,
                        block_size=128, num_layer=12, num_head=12, num_embd=768)
emailGenerator = TextGeneratorEmail(
    model_path=email_model_path)

@app.route('/api/gentext/marketing/email', methods=['POST'])
def emailMarketingAPI():
    data = request.get_json()

    prompt = data.get('prompt')
    max_len = data.get('max_len')
    temperature = data.get('temperature')
    print(prompt)
    print("temp", temperature)
    if temperature is None: temperature = 1.0
    if prompt == '' or prompt == None: prompt = ' '

    result, result_topk = emailGenerator.generate(
        inputs={"prompt": prompt}, stop_token_id=[emailGenerator.end_token_id], max_len=max_len, verbose=True, temperature=temperature or 1.0)
    prompt_len = len(emailGenerator.tokenizer.decode(
        emailGenerator.tokenizer.encode(prompt).ids))

    result = result.strip()
    result = result[prompt_len:]
    result_topk = result_topk[prompt_len:]

    response = {
        'text': result,
        'more': [{'text': result_topk, 'info':f'Top k sampling, k=1, temp={temperature}'}]
    }

    return jsonify(response)

vocab_pathv3 = 'src/models/transformers/bytelevel/models/vocab/chetna-md-16k-5f-vocab.json'
merges_pathv3 = 'src/models/transformers/bytelevel/models/vocab/chetna-md-16k-5f-merges.txt'
school_model_path = 'src/models/transformers/bytelevel/models/chetna_sm_12layer_12head_128block_1687671689.5749676_finetune_gptlike_newvocab_school_model'
TextGeneratorSchool = setup(vocab_path=vocab_pathv3, merges_path=merges_pathv3,
                        block_size=128, num_layer=12, num_head=12, num_embd=768)
schoolGenerator = TextGeneratorSchool(
    model_path=school_model_path)
@app.route('/api/gentext/marketing/school', methods=['POST'])
def marketingSchoolAPI():
    data = request.get_json()

    prompt = data.get('prompt')
    max_len = data.get('max_len')
    temperature = data.get('temperature')
    print(prompt)
    print("temp", temperature)
    if temperature is None: temperature = 1.0
    if prompt == '' or prompt == None: prompt = ' '

    result, result_topk = schoolGenerator.generate(
        inputs={"prompt": prompt}, stop_token_id=[schoolGenerator.end_token_id], max_len=max_len, verbose=True, temperature=temperature or 1.0)
    prompt_len = len(schoolGenerator.tokenizer.decode(
        schoolGenerator.tokenizer.encode(prompt).ids))

    result = result.strip()
    result = result[prompt_len:]
    result_topk = result_topk[prompt_len:]

    response = {
        'text': result,
        'more': [{'text': result_topk, 'info':f'Top k sampling, k=1, temp={temperature}'}]
    }

    return jsonify(response)

vocab_pathv3 = 'src/models/transformers/bytelevel/models/vocab/chetna-md-16k-5f-vocab.json'
merges_pathv3 = 'src/models/transformers/bytelevel/models/vocab/chetna-md-16k-5f-merges.txt'
restaurant_model_path = 'src/models/transformers/bytelevel/models/chetna_sm_12layer_12head_128block_1687669606.706952_finetune_gptlike_newvocab_restaurant_model'
TextGeneratorRestaurant = setup(vocab_path=vocab_pathv3, merges_path=merges_pathv3,
                        block_size=128, num_layer=12, num_head=12, num_embd=768)
restaurantGenerator = TextGeneratorRestaurant(
    model_path=restaurant_model_path)
@app.route('/api/gentext/marketing/restaurant', methods=['POST'])
def marketingResturantAPI():
    data = request.get_json()

    prompt = data.get('prompt')
    max_len = data.get('max_len')
    temperature = data.get('temperature')
    print(prompt)
    print("temp", temperature)
    if temperature is None: temperature = 1.0
    if prompt == '' or prompt == None: prompt = ' '

    result, result_topk = restaurantGenerator.generate(
        inputs={"prompt": prompt}, stop_token_id=[restaurantGenerator.end_token_id], max_len=max_len, verbose=True, temperature=temperature or 1.0)
    prompt_len = len(restaurantGenerator.tokenizer.decode(
        restaurantGenerator.tokenizer.encode(prompt).ids))

    result = result.strip()
    result = result[prompt_len:]
    result_topk = result_topk[prompt_len:]

    response = {
        'text': result,
        'more': [{'text': result_topk, 'info':f'Top k sampling, k=1, temp={temperature}'}]
    }

    return jsonify(response)


vocab_pathv3 = 'src/models/transformers/bytelevel/models/vocab/chetna-md-16k-5f-vocab.json'
merges_pathv3 = 'src/models/transformers/bytelevel/models/vocab/chetna-md-16k-5f-merges.txt'
story_model_path = 'src/models/transformers/bytelevel/models/chetna_sm_12layer_12head_128block_1687553886 (1).5368278_finetune_gptlike_story_model'
TextGeneratorStory = setup(vocab_path=vocab_pathv2, merges_path=merges_pathv2,
                        block_size=128, num_layer=12, num_head=12, num_embd=768)
storyGenerator = TextGeneratorStory(
    model_path=story_model_path)
@app.route('/api/gentext/story', methods=['POST'])
def storyAPI():
    print("story")
    data = request.get_json()

    prompt = data.get('prompt')
    max_len = data.get('max_len')
    temperature = data.get('temperature')
    print(prompt)
    print("temp", temperature)
    if temperature is None: temperature = 1.0
    if prompt == '' or prompt == None: prompt = ' '

    result, result_topk = storyGenerator.generate(
        inputs={"prompt": prompt}, stop_token_id=[storyGenerator.end_token_id], max_len=max_len, verbose=True, temperature=temperature or 1.0)
    prompt_len = len(storyGenerator.tokenizer.decode(
        storyGenerator.tokenizer.encode(prompt).ids))

    result = result.strip()
    result = result[prompt_len:]
    result = remove_offensive(result)
    result_topk = result_topk[prompt_len:]
    result_topk = remove_offensive(result_topk)

    response = {
        'text': result,
        'more': [{'text': result_topk, 'info':f'Top k sampling, k=1, temp={temperature}'}]
    }

    return jsonify(response)





def remove_offensive(text):
    bad_words = [ 'fuck' ]
    for w in bad_words:
        text = text.replace(w, '*'*len(w))
    return text

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4040, debug=True)

