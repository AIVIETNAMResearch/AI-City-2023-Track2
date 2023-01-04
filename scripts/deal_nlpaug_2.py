import json
import sys
import spacy
from spacy.matcher import Matcher

# sudo python3 -m spacy download en_core_web_sm
# before run this .py file
nlp = spacy.load("en_core_web_sm")


matcher = Matcher(nlp.vocab)
pattern = [
    [{"POS": "VERB", "op": "+"}, {"LOWER": "straight"}],
    [{"POS": "VERB", "op": "+"}, {"LOWER": "left"}],
    [{"POS": "VERB", "op": "+"}, {"LOWER": "right"}],
    [{"POS": "VERB", "op": "+"}, {"LOWER": "ahead"}],
    [{"LOWER": "stop"}],
    [{"LOWER": "stops"}],
          ]
matcher.add("motion-chunks", pattern)

# run for train and val
# train_path = 'data2021/train.json'
# # train_path = 'data2021/val.json'

path = ['data2022/train-tracks.json', 'data2022/train.json', 'data2022/val.json', 'data2022/test-queries.json']


def aug(train_path):
    with open(train_path) as f:
        train = json.load(f)

    track_ids = list(train.keys())
    for id_ in track_ids:
        new_text = ""
        for i, text in enumerate(train[id_]["nl"]):
            doc = nlp(text)

            # car aug
            for chunk in doc.noun_chunks:
                nb = chunk.text
                break
            train[id_]["nl"][i] = nb + '. ' + train[id_]["nl"][i]
            new_text += nb+'.'
            if i < 2:
                new_text += ' '

            # motion aug
            matches = matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                train[id_]["nl"][i] = train[id_]["nl"][i] + ' ' + span.text + '.'

        train[id_]["nl"].append(new_text)

    with open(train_path.split('.')[-2]+"_nlpaug_2.json", "w") as f:
        json.dump(train, f, indent=4)


for p in path:
    aug(p)
