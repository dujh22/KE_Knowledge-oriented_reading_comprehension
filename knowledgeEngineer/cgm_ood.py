import json
import requests
import re
import time
import string

url = "http://192.168.0.1"
header = {
    'Content-Type': 'application/json'
}

def ask_cgt(question):
    data = {
        'prompt': question
    }
    response = requests.post(url, headers=header, json=data)
    ans = response.json()
    ans2 = ans["response"]
    return ans2

with open("./ke/ood2.json", 'w', encoding='utf-8') as w:
    with open("./ke/ood_test2.json", 'r', encoding='utf-8') as ri:
        iid_list = json.load(ri)
        times = 0
        for item in iid_list:
            id = item['id']
            # title = item['title']
            passage = item['passage']
            question = item['question']['KoRC-H']

            try:
                passage_s = (re.findall(r'\[.*?\]', passage)[0]).strip("[]")
                str = f"{passage} What is {passage_s}? (if you don't know, please give me a guess answer)"
                mid_ans = ask_cgt(str)
                str2 = f"The {passage_s} is: {mid_ans}. {question} (The answer must be composed of several words, and must come from Wikidata, and the Wikidata ID of this answer can be checked. You just need to return me a few words, no more than 4, without punctuation)"
                ans = ask_cgt(str2)
            except:
                str2 = f"{passage} {question} (The answer must be composed of several words, and must come from Wikidata, and the Wikidata ID of this answer can be checked. You just need to return me a few words, no more than 4, without punctuation)"
                ans = ask_cgt(str2)

            print("\"{}\":\"{}\",\n".format(id, ans))

            w.write("\"{}\":\"{}\",\n".format(id, ans))
            w.flush()



