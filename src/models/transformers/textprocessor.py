

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import nltk
from typing import List

class TextProcessor:
    
    def __init__(self) -> None:
        self.tokenizer = get_tokenizer('basic_english')
        self.text = None
        self.text_list = None


    def load_data(self, path_to_txt):
        f = open(path_to_txt)
        self.text = f.read()
        f.close()

    def set_data_list(self, text_list: List):
        self.text_list = text_list

    def build_from_datalist(self, english_only=False, expand_words=False):
        self.tokens_list = None
        def yield_tokens(data_list):
            for tl in data_list:
                yield self.tokenizer(tl)
        if expand_words:
            self.text_list = [[ expand_contractions(t) for t in ti ] for ti in self.text_list]

        if english_only:
            nltk.download('words')

            # Set up a word filter with English words
            english_words = set(nltk.corpus.words.words())
            punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

            def is_english_word(word):
                return word.lower() in english_words or word in punctuations
            
            self.tokens_list = [self.tokenizer(t) for t in self.text_list]
            
        else:
            self.tokens_list = [self.tokenizer(t) for t in self.text_list]
            self.vocab = build_vocab_from_iterator(yield_tokens(self.text_list))
            self.token_ids = [self.encode(t) for t in self.tokens_list]


    def build(self, path=None, english_only=not True):

        def yield_tokens(data):
            yield self.tokens

        text = self.text
        if english_only:
            nltk.download('words')

            # Set up a word filter with English words
            english_words = set(nltk.corpus.words.words())
            punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

            def is_english_word(word):
                return word.lower() in english_words or word in punctuations
            
            self.tokens = self.tokenizer(text)
            self.tokens = [tok for tok in self.tokens if is_english_word(tok)]  #filter out all the non english words
        else:
            if text is not None:
                self.tokens = self.tokenizer(text)

        if path is None:
            self.vocab = build_vocab_from_iterator(yield_tokens([self.text]), specials=['<unk>'])
            # self.vocab.set_default_index(-1)
            self.vocab.set_default_index(self.vocab['<unk>'])
        else:
            self.vocab = torch.load(path)
            # self.vocab.set_default_index(self.vocab['<unk>'])

        if text is not None:    
            self.token_ids = torch.tensor(self.encode(self.tokens), dtype=torch.long)
           

    def get_vocab(self):
        return self.vocab.get_itos()
    
    def get_vocab_size(self):
        return len(self.get_vocab())
    
    def save(self, path='vocab.pth'):
        torch.save(self.vocab, path)
        print("vocab saved at", path)

    def encode(self, tokens):
        return self.vocab(tokens)
    
    def decode(self, ids):
        return " ".join([self.vocab.get_itos()[id] for id in ids ])
    
# textProcessor = TextProcessor()
# textProcessor.load_data('data/dream_quotes.txt')
# # textProcessor.build()
# textProcessor.build(english_only=False)

def expand_contractions(text):
    # Dictionary of common contractions and their expanded form
    contractions = {"ain't": "am not", "aren't": "are not", "can't": "cannot",
                    "could've": "could have", "couldn't": "could not", "didn't": "did not",
                    "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                    "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                    "he'll": "he will", "he's": "he is", "I'd": "I would", "I'll": "I will",
                    "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would",
                    "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam",
                    "might've": "might have", "mightn't": "might not", "must've": "must have",
                    "mustn't": "must not", "needn't": "need not", "oughtn't": "ought not",
                    "shan't": "shall not", "she'd": "she would", "she'll": "she will",
                    "she's": "she is", "should've": "should have", "shouldn't": "should not",
                    "that's": "that is", "there's": "there is", "they'd": "they would",
                    "they'll": "they will", "they're": "they are", "they've": "they have",
                    "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are",
                    "we've": "we have", "weren't": "were not", "what'll": "what will",
                    "what're": "what are", "what's": "what is", "what've": "what have",
                    "where's": "where is", "who'd": "who would", "who'll": "who will",
                    "who're": "who are", "who's": "who is", "who've": "who have",
                    "won't": "will not", "would've": "would have", "wouldn't": "would not",
                    "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have"}
    
    # Replace contractions with their expanded form
    for contraction, expanded_form in contractions.items():
        text = text.replace(contraction, expanded_form)
        
    return text
