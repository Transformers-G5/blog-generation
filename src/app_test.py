from flask import Flask, send_from_directory, request, flash, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


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
        'gen_text': result,
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
        'gen_text': result,
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
        'gen_text': result,
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
        'gen_text': result,
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
        'gen_text': result,
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



