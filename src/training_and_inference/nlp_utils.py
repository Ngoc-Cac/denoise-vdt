import regex as re
import unicodedata
import underthesea as uts


def normalise_vietnamese(text: str) -> str:
    """
    Normalise a text in Vietnamese. This will normalise the text
    into the Unicode NFC and remove any punctuation.

    Then, the text is further normalised by removing space and fix accent
    placements with `underthesea`, before being tokenised by `underthesea`.

    :param str text: The text to normalise.
    :return: The normalised text.
    :rtype: str
    """
    nfc_text = unicodedata.normalize('NFC', text)
    return uts.word_tokenize(
        re.sub(r"\p{P}+", "", uts.text_normalize(nfc_text).casefold()),
        format="text"
    )


def preprocess_vn_text(texts: list[str]) -> list[str]:
    return [normalise_vietnamese(text) for text in texts]
