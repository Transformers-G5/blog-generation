from transformers import pipeline


class TextGenerator:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.generator = pipeline('text-generation', model=self.model_path)
        self.generatedText = ""

    def generateText(self, prompt="", max_length=10):
        print(">> Generating text")
        self.generatedText = self.generator(
            prompt, max_length=max_length, do_sample=True, temperature=0.9)
        self.generatedText = self.generatedText[0]['generated_text']
        print(">> Text generated for prompt :", prompt)
        return self.generatedText

    def saveGeneratedText(self, fileName="output.txt"):
        with open(fileName, 'w') as f:
            f.writelines(self.generatedText)


if __name__ == "__main__":
    textGen = TextGenerator("./gpt-neo-125M")
    prompt = "The uprising of apple iphone"
    textGen.generateText(prompt, 500)
    textGen.saveGeneratedText()
