import nltk
# nltk.download()
# from subprompt_template import SubTemplateDataLoader
import requests
from googleapiclient.discovery import build
import pprint
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

path_to_model = os.path.join(
    os.getcwd(), "src/models", "t5-base-finetuned-common_gen")

tokenizer = AutoTokenizer.from_pretrained(path_to_model)
model = AutoModelForSeq2SeqLM.from_pretrained(path_to_model)


my_api_key = "AIzaSyB0bwYCFOqsz9guwKHKAPKHOvwZ-oUqTSY"
my_cse_id = "c0dfdebfea2ca432f"

from spacy.symbols import *
import spacy   
from spacy.matcher import Matcher
from spacy.util import filter_spans

class SubPromptgenerator:
    def __init__(self, num_prompts=4) -> None:
        self.num_prompts = num_prompts
        self.puncs = ["?", "!", "."]
        self.q_pres = ["How", "When", "What", "Where", "Is", "Does", "Do", "Is it"]

        self.nlp_noun = spacy.load('en_core_web_sm') 


        # self._nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")
        # k2t = pipeline("k2t")
        # k2tbase = pipeline("k2t-base")

    def nlp(self, words, max_length=32):
        input_text = " ".join(words)
        features = tokenizer([input_text], return_tensors='pt')
        output = model.generate(input_ids=features['input_ids'],
                                attention_mask=features['attention_mask'],
                                max_length=max_length)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        return output

    def generate(self, prompt):
        '''
            generates subprompts
        '''
        # find nouns
        # nouns = self.__findNouns(prompt)
        # print(nouns)
        # find contexts
        # find probabilities for pre defined subprompt templates
        # create subpropmts using the most probable n subpropmts

        # subs = self.__getSubpromptsWeb(prompt )
        subs = self.__generateSubPromptsLocal(prompt)

        return subs

    def __generateSubPromptsLocal(self, prompt):
        k = list(prompt.split(" "))
        k_qs = []

        k_nouns = []   #list to store nouns from the prompt
        nouns_map = self.__findNouns(prompt)
        for key in nouns_map.keys():
          k_nouns.append(key)
        
        # print(k_nouns, "nouns")


        k_qs_noun = []   #list to store augmanted text using nouns

        for i in self.q_pres:
          k_qs_noun.append([i] + k_nouns + ["?"])


        for i in self.q_pres:
            k_qs.append([i] + k + ["?"])

        sub_qks = []
        for i in k_qs:
            sub_qks.append(self.nlp(i))
        for i in k_qs_noun:
          sub_qks.append(self.nlp(i))

        sub_qks = self.__removeduplicate([self.nlp(k)] + sub_qks )

        sub_qks = list(map(lambda x: x + "?", sub_qks))


        sub = sub_qks


        
        return sub

 
    def __findNouns(self, sentence):
        # is_noun = lambda pos: pos[:2] == 'NN'
        # tokenized = nltk.word_tokenize(prompt)
        # nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        doc = self.nlp_noun(sentence)
        np_labels = set([nsubj, nsubjpass, dobj, iobj, pobj])
        nounsMap = {}
        for word in doc:
          if word.dep in np_labels:
            nounsMap[word.text] = word.dep_
            
        return nounsMap
        return nouns
    def __removepunc(self, sentence):
      punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
 
      # Removing punctuations in string
      # Using loop + punctuation string
      for ele in sentence:
          if ele in punc:
              sentence = sentence.replace(ele, "")
      return sentence

    def __removeduplicate(self, str_list):

      str_list =list(map( self.__removepunc, str_list))
      #remove all the spaces
      str_list = list(map(lambda x: x.strip(), str_list))
      return list(set(str_list))


    def __google_search(self, search_term, api_key, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return res['items']

    def __getSubpromptsWeb(self, search_query):
        sub_prompts_web = []
        results = self.__google_search(
            search_query, my_api_key, my_cse_id, num=self.num_prompts)
        for result in results:
            res = result['title']
            res = self.__preprocessSubs(res)
            sub_prompts_web.append(res)

        return sub_prompts_web

    def __preprocessSubs(self, sentense):
        res = ''
        sentense = sentense[::-1]
        for i in range(len(sentense)):
            # if sentense[i] == "." or sentense[i] == " ":
            #     continue
            # else:
            #     res = sentense[i:][::-1]
            #     break
            if sentense[i] in self.puncs:
                res = sentense[i:][::-1]
                break

        return res

    def __findContexts(self, propmt):
        return []

    def __findProbablitiesOfSubTemplates(self, prompt, nouns, contexts):
        return {}

    def __createSubpromptsFromTemplates(self, nouns, context, template_probs):
        return []


if __name__ == "__main__":
    prompt = "Is iphone better than andoid?"
    spg = SubPromptgenerator()
    sub = spg.generate(prompt=prompt)
    print(sub)
