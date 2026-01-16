'''A function to clean the text, bascically removes noise such as links,@,# and other special characters
from the tweet texts'''

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+','',text)   #removing url's/replacing them with ''
    text = re.sub(r'#','',text)   #removing hashtags/ replacing them with ''
    text = re.sub(r'@\w+','',text)  #removing mentions/replacing them with ''
    text = re.sub(r'\d+','',text)  #removing numbers/ replacing them with ''
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)  #removing special characters

    return text
