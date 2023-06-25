import json
import requests
import re
import time
import string

url = "http://192.168.0.1"

def ask_cgt(question):
    data = {
        'user': question
    }
    response = requests.post(url, data=data, verify=False)
    if response.status_code != 200:
        ans2 = "Unknown"
    else:
        ans = response.json()
        ans2 = ans["choices"][0]["message"]["content"]
    return ans2

with open("./ke/ood.json", 'w', encoding='utf-8') as w:
    with open("./ke/ood_test.json", 'r', encoding='utf-8') as ri:
        iid_list = json.load(ri)
        times = 0
        for item in iid_list:
            id = item['id']
            # title = item['title']
            passage = item['passage']
            question = item['question']['KoRC-H']

            passage_s = (re.findall(r'\[.*?\]', passage)[0]).strip("[]")

            str = f"{passage} What is {passage_s}? (if you don't know, please give me a guess answer)"
            mid_ans = ask_cgt(str)
            start_time = time.time()

            str2 = f"The {passage_s} is: {mid_ans}. {question} (The answer must be composed of several words, and must come from Wikidata, and the Wikidata ID of this answer can be checked. You just need to return me a few words, no more than 4, without punctuation)"
            print(str2)

            ans = ask_cgt(str2)
            print(ans)

            if ans[-1] == '.':
                ans = ans[:-1]
            print(ans)

            if len(ans) > 6:
                ans = "Unknown"

            end_time = time.time()

            print("\"{}\":\"{}\",\n".format(id, ans))

            w.write("\"{}\":\"{}\",\n".format(id, ans))
            w.flush()

            delta_time = end_time - start_time
            print(delta_time)
            time.sleep(60 - delta_time)


