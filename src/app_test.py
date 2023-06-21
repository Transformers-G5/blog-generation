from flask import Flask, send_from_directory, request, flash, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


from models.transformers.bytelevel.model_v2 import setup

vocab_pathv2 = 'src/models/transformers/bytelevel/models/vocab/chetna-sm-5k-3f-vocab.json'
merges_pathv2 = 'src/models/transformers/bytelevel/models/vocab/chetna-sm-5k-3f-merges.txt'
blog_model_path = 'src/models/transformers/bytelevel/models/chetna_sm_12layer_6head_64block_1687334767.7194476_blog_model'
TextGenerator = setup(vocab_path=vocab_pathv2, merges_path=merges_pathv2,
                        block_size=64, num_layer=12, num_head=6, num_embd=768)
chatAllTGV2 = TextGenerator(
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

    result, result_topk = chatAllTGV2.generate(
        inputs={"prompt": prompt}, stop_token_id=[chatAllTGV2.end_token_id], max_len=max_len, verbose=True, temperature=temperature or 1.0)
    prompt_len = len(chatAllTGV2.tokenizer.decode(
        chatAllTGV2.tokenizer.encode(prompt).ids))

    result = result.strip()
    result = result[prompt_len:]
    result_topk = result_topk[prompt_len:]

    response = {
        'gen_text': result,
        'more': [{'text': result_topk, 'info':f'Top k sampling, k=1, temp={temperature}'}]
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4040, debug=True)