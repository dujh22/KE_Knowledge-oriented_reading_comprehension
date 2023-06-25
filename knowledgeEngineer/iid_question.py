import json
import re

entity_list = []
with open("./ke/iid_question_entity.txt", 'r', encoding='utf-8') as r:
    for line in r:
        xu, entity = line.split("	")
        if entity == '\n':
            entity = "Unknown"
        else:
            entity = entity.strip()
        entity_list.append(entity)

last_title = "first"
xu = -1

with open("./ke/iid_question.json", 'w', encoding='utf-8') as w:
    with open("ke/iid_test.json", 'r', encoding='utf-8') as ri:
        iid_list = json.load(ri)
        for item in iid_list:
            id = item['id']

            title = item['title']
            if title != last_title:
                last_title = title
                xu = xu + 1

            passage = item['passage']
            passage_s = (re.findall(r'\[.*?\]', passage)[0]).strip("[]")

            question = item['question']['KoRC-H']
            question = question.replace(passage_s, entity_list[xu])

            print("\"{}\":\"{}\",\n".format(id, question))
            w.write("\"{}\":\"{}\",\n".format(id, question))
            w.flush()


