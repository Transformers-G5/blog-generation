from models.transformers.model import Main

main = Main()
# main.config(encoder_dict_path='./models/inspirational_encoder.pkl', decoder_dict_path='./models/inspirational_decoder.pkl')
tg = main.build(model_path='production/blog/models/blogs_v8.1_loss_0.07_model',vocab_path='production/blog/models/blogs_vocab_v8.1_0.07' )

result = tg.generate(inputs={"prompt":"happiness"}, max_len=500, verbose=True)

print(result)