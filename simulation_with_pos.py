import os
import json
import numpy as np
from module.preprocessing import get_index_list_of_sentences
from keras.models import model_from_json
from keras.optimizers import Adam
from keras_contrib.layers import CRF
from module.test import get_arg_list
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

import warnings
warnings.filterwarnings("ignore")

input_sentence = input('Input your sentence: ')

# config variable
CONFIG_ROOT = 'CONFIG'
CONFIG_NAME = 'POS_TAG_CONFIG.json'
CONFIG_WORDS = 'WORDS_CONFIG.json'

CONFIG_PATH = os.path.join(CONFIG_ROOT, CONFIG_NAME)
WORDS_PATH = os.path.join(CONFIG_ROOT, CONFIG_WORDS)

with open(CONFIG_PATH) as config:
    config_dic = json.load(config)

with open(WORDS_PATH) as words_path:
    words_config = json.load(words_path)

max_sentence_length = config_dic['max_sentence_length']
index_to_word_dic = words_config['index_to_word_dic_with_pos']
vocab_size = words_config['vocab_size_with_pos']
ner_to_index_dic = words_config['ner_to_index_dic']
B_PER_index = ner_to_index_dic['B-PER']
I_PER_index = ner_to_index_dic['I-PER']
n_labels = len(ner_to_index_dic)
word_to_index_dic = {tuple(value): int(key) for key, value in index_to_word_dic.items()}


def padding(x):
    if len(x) >= max_sentence_length:
        return x[:max_sentence_length]
    else:
        return x + [0] * (max_sentence_length - len(x))


input_sentence_list = pos_tag(word_tokenize(input_sentence))
input_sentence_list = list(map(lambda x: (x[0], x[1]), input_sentence_list))
input_sentence_list = [(word, pos)for word, pos in input_sentence_list]

input_sentence_with_index = get_index_list_of_sentences(
    [[(word, "O") for word in input_sentence_list]], word_to_index_dic)
final_input = np.array([padding(x) for x in input_sentence_with_index])

# 모델 불러오기
MODEL_ROOT = 'model'
MODEL_JSON = 'model_with_pos.json'
MODEL_H5 = 'model_with_pos.h5'


MODEL_JSON_PATH = os.path.join(MODEL_ROOT, MODEL_JSON)
MODEL_H5_PATH = os.path.join(MODEL_ROOT, MODEL_H5)

with open(MODEL_JSON_PATH, 'r') as f:
    loaded_model_json = f.read()

loaded_model = model_from_json(loaded_model_json, custom_objects={'CRF': CRF})
loaded_model.load_weights(MODEL_H5_PATH)

crf = CRF(n_labels)

loaded_model.compile(loss=crf.loss_function,
                     optimizer=Adam(0.001), metrics=[crf.accuracy])

# 예측하기
input_predict = loaded_model.predict(final_input)

len_input = len(input_sentence_list)
input_predict_word_index = get_arg_list(input_predict)[:len_input]


# 출력
division_length = 80
print("result  ".ljust(division_length, '-'))
result = 'Yes' if B_PER_index in input_predict_word_index or I_PER_index in input_predict_word_index else "No"
print("Does this sentence have any named entity of persons?", result)
if result == 'Yes':
    print()
    for word, ner_index in zip(input_sentence_list, input_predict_word_index):
        if ner_index == B_PER_index or ner_index == I_PER_index:
            print(word[0], '<PER>')
        else:
            print(word[0])
print("-" * division_length)
print()
print()

print(input_predict_word_index)
print(ner_to_index_dic)