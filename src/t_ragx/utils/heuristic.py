import re

import regex
import unicodedata


def lang_detect(text):
    if text is None:
        return None

    ja_text = r"[\p{Katakana}\p{Hiragana}]"
    en_text = r"[a-zA-Z]"
    zh_text = r"\p{han}"

    text_len_dict = {
        'ja': len(regex.findall(ja_text, text)),
        'en': len(regex.findall(en_text, text)),
        'zh': len(regex.findall(zh_text, text))
    }

    max_key = max(text_len_dict, key=text_len_dict.get)

    # # filtering?
    # total_len = len(text)
    # if text_len_dict[max_key] / total_len < 0.5:
    #     return None
    return max_key


def clean_text(text):
    return unicodedata.normalize('NFKD', text).strip()


def is_date(text):
    # date_format_list = ['%Y年%m月%d', '%Y年%m月%d日', '%Y年', '%m月%d日', '%d日', '%m月']
    date_format_list = ['\d{1,}年\d{1,}月', '\d{1,}年\d{1,}月\d{1,}日', '\d{1,}年', '\d{1,}月\d{1,}日', '\d{1,}日',
                        '\d{1,}月']
    for date_format in date_format_list:
        # if not pd.isna(pd.to_datetime(text, format=date_format, errors='coerce')):
        if re.match('^' + date_format + '$', text) is not None:
            return True
    return False


def is_number(text):
    return re.match("^\d{1,}\.?\d*$", text) is not None


# assert is_number("54564.564") == True
# assert is_number("24778") == True
# assert is_number("24.7.78") == False
# assert is_number("24..778") == False
# assert is_number("1a24778") == False
# assert is_number("a124778") == False
# assert is_number("a124778f") == False


def is_noise(text):
    """
    many text in wiki or web crawled data is just number or date, which is not very useful as memory or glossary
    """
    if is_number(text):
        return True

    if is_date(text):
        return True
    return False
