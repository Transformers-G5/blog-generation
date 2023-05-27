import random


def remove_start_end(text: str) -> str:
    if '[end]' in text:
        # remove the [star] and [end] tokens
        text = text.replace('[start]', '')
    return text


def capitalize_first(text: str, splitings=['. ']) -> str:
    if len(text.strip()) <= 1:
        return text
    for s in splitings:
        text = text.split(s)
        text = [t.lstrip() for t in text]
        # handle empty or less than 2 len strings
        text = [t[0].upper() + t[1:] if len(t) > 1 else t for t in text]
        text = s.join(text)

    return text


def strip_contractions(text: str) -> str:
    text = text.replace(" ' ", "'")
    text = text.replace("' ", "'")
    return text


def decorate_response(text: str) -> str:
    prefixes = ['Sure!', "Absolutely!",
                "Sure, I'd be happy to help you out!", ]
    text = f"<b>{prefixes[random.randint(0, len(prefixes) - 1)]}</b></br>" + \
        text + ""
    return text
