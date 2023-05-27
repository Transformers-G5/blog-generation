from model import Main


main = Main()
main.config(encoder_dict_path='src/models/transformers/inspirational_encoder.pkl', decoder_dict_path='src/models/transformers/inspirational_decoder.pkl')
tg = main.build(model_path='src/models/transformers/inspirational_char_loss_.26_')


result = tg.generate(inputs={"prompt":"What is happiness?"}, max_len=500, verbose=True)

print(result)