from utils.postprocessors.common_postprocessors import capitalize_first, strip_contractions


def simple_letter_formater(text: str, placeholders={}) -> str:
    text = text.replace('[linebreak]', ' </br> ')
    # fill the placeholders
    for placeholder_name, placeholder_value in placeholders.items():
        if placeholder_value is not None:
            text = text.replace(placeholder_name, placeholder_value)
    text = capitalize_first(text, splitings=[". ", "</br>"])

    text = strip_contractions(text)
    text = text.replace(' ,', ',')
    text = text.replace(' . ', '. ')
    return text


def make_bold(text):
    return f"<b>{text}</b>"


def createLoveLetterPlaceHolder(to_name=None, from_name=None):
    return {
        "[partner ' s name]":  make_bold(to_name),
        '[your name]': make_bold(from_name)
    }
