from googletrans import Translator
from tqdm import tqdm
import spacy
from spacy.matcher import Matcher
import collections

color_labels = {"white": ["white", "whitish"],
                "black": ["black", "dark", "drak"],
                "gray": ["gray", "silver", "grey"],
                "red": ["red", "scarlet", "reddish", "maroon"],
                "blue": ["blue", "purple"],
                "green": ["green"],
                "brown": ["brown", "beige", "tan"],
                "yellow": ["yellow", "golden", "gold", "orange"]}

type_labels = {"sedan": ["sedan", "suburu", 'car', "ford mustang", "coupe"],
               "truck": ["pickup", "truck", "wagon"],
               "suv": ["suv", "jeep", "mpv"],
               "van": ["van", "minivan"],
               "bus": ["bus"],
               "hatchback": ["hatchback"]}

motion_labels = {
    "straight": ["straight", "ahead", "forward", "down"],
    "left": ["left"],
    "right": ["right"],
    "stops": ["stop", "stops", "stopped"]
}


def back_translate_aug(text, lang='vi'):
    translator = Translator()
    return translator.translate(translator.translate(text, dest=lang).text, dest='en').text


def augmentation(data, lang='vi', n=1):
    augmented_text = data['nl']
    for text in data['nl']:
        for _ in range(n):
            augmented_text.append(back_translate_aug(text, lang=lang))

    data['augmented_nl'] = augmented_text

    return data
    

def extract_color_from_text(data):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    pattern = [{'POS': 'ADJ'}, {'POS': 'NOUN'}]
    matcher.add('collor-chunks', pattern)

    colors = []

    for idx, text in enumerate(data['nl']):
        doc = nlp(text)
        matches = matcher(doc)
        
        for match_id, start, end in matches:
            span = doc[start:end]
            for key, val in color_labels.items():
                if span.text.lower() in val:
                    colors.append(key)
        
    if len(colors) > 0:
        counter = collections.Counter(colors)
        color = counter.most_common(1)[0][0]
        data['color'] = color
    else:
        data['color'] = 'unknown'

    return data

def extract_vehicle_type_from_text(data):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    pattern = [[{'POS': 'NOUN'}]]
    matcher.add('vehicle-chunks', pattern)

    vehicle_types = []

    for idx, text in enumerate(data['nl']):
        doc = nlp(text)
        matches = matcher(doc)
        
        for match_id, start, end in matches:
            span = doc[start:end]
            for key, val in type_labels.items():
                if span.text.lower() in val:
                    vehicle_types.append(key)
        
    if len(vehicle_types) > 0:
        counter = collections.Counter(vehicle_types)
        vehicle_type = counter.most_common(1)[0][0]
        data['vehicle_type'] = vehicle_type
    else:
        data['vehicle_type'] = 'unknown'

    return data

def extract_motion_from_text(data):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    pattern = [[{'POS': 'VERB'}]]
    matcher.add('motion-chunks', pattern)

    motions = []

    for idx, text in enumerate(data['nl']):
        doc = nlp(text)
        matches = matcher(doc)
        
        for match_id, start, end in matches:
            span = doc[start:end]
            for key, val in motion_labels.items():
                if span.text.lower() in val:
                    motions.append(key)
        
    if len(motions) > 0:
        counter = collections.Counter(motions)
        motion = counter.most_common(1)[0][0]
        data['motion'] = motion
    else:
        data['motion'] = 'unknown'

    return data