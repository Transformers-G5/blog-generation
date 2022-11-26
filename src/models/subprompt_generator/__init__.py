import nltk
# nltk.download()
# from subprompt_template import SubTemplateDataLoader
import requests
from googleapiclient.discovery import build
import pprint

my_api_key = "AIzaSyB0bwYCFOqsz9guwKHKAPKHOvwZ-oUqTSY"
my_cse_id = "c0dfdebfea2ca432f"

class SubPromptgenerator:
    def __init__(self, num_prompts=4) -> None:
        self.num_prompts = num_prompts
        self.puncs = ["?", "!", "."]
        pass

    def generate(self, prompt):
        '''
            generates subprompts

        '''

        
        #find nouns
        # nouns = self.__findNouns(prompt)
        # print(nouns)
        #find contexts
        #find probabilities for pre defined subprompt templates
        #create subpropmts using the most probable n subpropmts

        subs = self.__getSubpromptsWeb(prompt )

        return subs
    
    def __findNouns(self, prompt):
        is_noun = lambda pos: pos[:2] == 'NN'
        tokenized = nltk.word_tokenize(prompt)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        return nouns

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