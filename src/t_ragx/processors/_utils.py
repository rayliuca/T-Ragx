import base64
import json
import os.path
import pathlib
import tempfile
import urllib.request
from hashlib import md5

import requests

from ..utils.heuristic import clean_text


def serialize_str(s):
    return json.dumps(s, ensure_ascii=False)


def en_text_search(text, keyword):
    if len(keyword) > len(text):
        return False
    text = text.casefold()
    keyword = keyword.casefold()
    if f" {keyword} " in text:
        return True
    elif text == keyword:
        return True
    elif len(keyword) - 1 > len(text) and text[:len(keyword) + 1] == keyword + " ":
        return True
    elif len(keyword) - 1 > len(text) and text[-len(keyword) + 1:] == " " + keyword:
        return True
    return False


# heuristic glossary retrieval
def get_glossary(text, glossary_index, max_k=10, lang_code='en', source_lang='ja'):
    text = clean_text(text)
    out_dict = {}
    count = 0
    for k in glossary_index:
        if lang_code not in glossary_index[k]:
            continue
        if (k in text and source_lang != 'en') or \
                (source_lang == 'en' and en_text_search(text, k)):
            skip_flag = False
            # check for glossary word being a component of a longer glossary word
            for ek in out_dict:
                if k.casefold() in ek.casefold():
                    skip_flag = True
                    break
            if skip_flag:
                continue

            out_dict[k] = glossary_index[k][lang_code].tolist()
            count += 1
            if count >= max_k:
                break

    return out_dict


def get_http_file_id(url):
    response = requests.head(url)
    # use ETag if available
    if 'ETag' in response.headers:
        return response.headers['ETag'].replace('"', "")

    # use encoded url path if ETag is not available
    return md5(base64.urlsafe_b64encode(url.encode())).hexdigest()


def file_cacher(file_path, tempfolder=None):
    """
    If the input file_path is a http url, cache the file (by ETag if possible) to local tempfolder

    Args:
        file_path:
        tempfolder:

    Returns:

    """
    if tempfolder is None:
        tempfolder = tempfile.gettempdir() + "/t_ragx"
        pathlib.Path(tempfolder).mkdir(parents=True, exist_ok=True)
    out_path = file_path
    if "http" in file_path:
        file_id = get_http_file_id(file_path)
        file_extension = pathlib.Path(file_path).suffix
        out_path = f"{tempfolder}/{file_id}{file_extension}"

        if not os.path.isfile(out_path):
            urllib.request.urlretrieve(file_path, out_path)

    return out_path
