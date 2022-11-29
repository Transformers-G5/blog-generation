from transformers import pipeline


class TextGenerator:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.generator = pipeline('text-generation', model=self.model_path)
        self.generatedText = ""
        self.intro = ""
        self.paras = []

    def generateText(self, prompt="", max_length=10):
        print(">> Generating text")
        self.generatedText = self.generator(
            prompt, max_length=max_length, do_sample=True, temperature=0.9)
        self.generatedText = self.generatedText[0]['generated_text']
        print(">> Text generated for prompt :", prompt)
        return self.generatedText

    def __generate_intro(self, prompt, max_length=200):
        self.intro = self.generator(
            prompt, max_length=max_length, do_sample=True, temperature=0.9)
        self.intro = self.intro[0]['generated_text']

    def __generate_para(self, prompt, max_length=300):
        para = self.generator(prompt, max_length=max_length,
                              do_sample=True, temperature=0.9)
        para = para[0]['generated_text']
        self.paras.append(para)

    def generateBlog(self, prompt="", subPrompts=[]):
        print(">> Generating blog")
        self.__generate_intro(prompt, max_length=200)
        for sprompt in subPrompts:
            self.__generate_para(sprompt, max_length=300)
        return self.intro, self.paras

    def clean(self):
        self.intro = ""
        self.paras = []

    def saveGeneratedText(self, fileName="output.txt"):
        with open(fileName, 'w') as f:
            f.writelines(self.generatedText)


if __name__ == "__main__":
    textGen = TextGenerator("./gpt-neo-125M")
    prompt = "The uprising of apple iphone"
    textGen.generateText(prompt, 500)
    textGen.saveGeneratedText()
