import json
class SubTemplateDataLoader:
    def __init__(self) -> None:
        pass
    def loadByAmount(self, amount=None):
        if amount is None:
            f = open("./templates.json")
            data = json.load(f)
            return data
        return data[:amount]





##testing 
if __name__ == '__main__':
    stdl = SubTemplateDataLoader()
    data = stdl.loadByAmount()
    print(data[0]['template'])