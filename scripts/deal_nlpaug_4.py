# from preprocessing.transforms import BackTranslateAug
import json
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel
import os

path = ['data/AIC23_Track2_NL_Retrieval/data/train_nlpaug.json']


class BackTranslateAug(object):
    def __init__(self, first_model_name='Helsinki-NLP/opus-mt-en-fr', second_model_name='Helsinki-NLP/opus-mt-fr-en') -> None:
        self.first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)
        self.first_model  = MarianMTModel.from_pretrained(first_model_name)

        self.second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)
        self.second_model = MarianMTModel.from_pretrained(second_model_name)
    
    def format_batch_texts(self,language_code, batch_texts):
        formatted_batch = [">>{}<< {}".format(language_code, text) for text in batch_texts]

        return formatted_batch
    
    def translate(self, batch_texts, model, tokenizer, language="fr"):
        # Prepare the text data into appropriate format for the model
        formated_batch_texts = self.format_batch_texts(language, batch_texts)
        
        # Generate translation using model
        translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))

        # Convert the generated tokens indices back into text
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        
        return translated_texts

    def __call__(self, original_texts, language="fr"):
        translated_text = self.translate(original_texts, self.first_model, self.first_model_tkn)
        back_translate_text = self.translate(translated_text, self.second_model, self.second_model_tkn)
        return back_translate_text


def aug(train_path):
    with open(train_path) as f:
        train = json.load(f)
    
    text_aug = BackTranslateAug()
    track_ids = list(train.keys())
    for id_ in tqdm(track_ids, total=len(track_ids)):
        aug_texts = text_aug(train[id_]['nl'])
        train[id_]['nl'] = aug_texts

    with open(train_path.split('.')[-2]+"_nlpaug_4.json", "w") as f:
        json.dump(train, f, indent=4)

for p in path:
    aug(p)