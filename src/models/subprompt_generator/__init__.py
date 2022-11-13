import nltk
nltk.download()
from subprompt_template import SubTemplateDataLoader
class SubPromptgenerator:
    def __init__(self, num_prompts=4) -> None:
        self.num_prompts = num_prompts
        pass

    def generate(self, prompt):
        #find nouns
        nouns = self.__findNouns(prompt)
        print(nouns)
        #find contexts
        #find probabilities for pre defined subprompt templates
        #create subpropmts using the most probable n subpropmts

        pass
    
    def __findNouns(self, prompt):
        is_noun = lambda pos: pos[:2] == 'NN'
        tokenized = nltk.word_tokenize(prompt)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        return nouns

    def __findContexts(self, propmt):
        return []
    
    def __findProbablitiesOfSubTemplates(self, prompt, nouns, contexts):
        return {}

    def __createSubpromptsFromTemplates(self, nouns, context, template_probs):
        return []

    
if __name__ == "__main__":
    prompt = "Is iphone better than andoid?"
    spg = SubPromptgenerator()
    spg.generate(prompt=prompt)