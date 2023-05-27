import re

tags_pattern = re.compile('<.*?>')

def remove_tags(text: str) -> str:
    text = re.sub(tags_pattern, '', text)
    return text