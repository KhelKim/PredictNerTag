import os
import json
import numpy as np
import pandas as pd
from module.EDA import get_word_count_dic
from module.preprocessing import *


def padding(x):
    if len(x) >= max_sentence_length:
        return x[:max_sentence_length]
    else:
        return x + [0] * (max_sentence_length - len(x))


# 데이터 주소 정보
DATA_ROOT = 'dataset'
DATA_NAME = 'train_clean.csv'
TEST_FILE_NAME = 'test_clean.csv'

CONFIG_ROOT = 'CONFIG'
CONFIG_NAME = 'POS_TAG_CONFIG.json'

DATA_PATH = os.path.join(DATA_ROOT, DATA_NAME)
TEST_FILE_PATH = os.path.join(DATA_ROOT, TEST_FILE_NAME)
CONFIG_PATH = os.path.join(CONFIG_ROOT, CONFIG_NAME)

# 데이터 불러오기
data = pd.read_csv(DATA_PATH)
test = pd.read_csv(TEST_FILE_PATH)

with open(CONFIG_PATH) as config:
    config_dic = json.load(config)

max_sentence_length = config_dic['max_sentence_length']
min_word_count = config_dic["min_word_count"]

# 1. 소문자화
# data['word'] = data['word'].str.lower()

# 2. df를 리스트 형태로 변환
text = get_text_list(data, 'sentence_index', pos_tag=False)
text_with_pos = get_text_list(data, 'sentence_index')

# 3. word count dic 만들기
word_count_dic = get_word_count_dic(text)
word_count_dic_with_pos = get_word_count_dic(text_with_pos)

# 4. word to index dic, index to word dic 만들기
word_to_index_dic = get_word_to_index_dic(text, word_count_dic, min_word_count, pos_tag=False)
word_to_index_dic_with_pos = get_word_to_index_dic(text_with_pos, word_count_dic_with_pos, min_word_count)

index_to_word_dic = {value: key for key, value in word_to_index_dic.items()}
index_to_word_dic_with_pos = {value: key for key, value in word_to_index_dic_with_pos.items()}

vocab_size = len(word_to_index_dic)
vocab_size_with_pos = len(word_to_index_dic_with_pos)

# 5. ner to index dic, index to ner dic 만들기
ner_to_index_dic = get_ner_to_index_dic(text)
index_to_ner_dic = {value: key for key, value in ner_to_index_dic.items()}

# 6. sentences with index, ner with index 리스트 만들기
sentences_with_index = get_index_list_of_sentences(text, word_to_index_dic, pos_tag=False)
sentences_with_index_with_pos = get_index_list_of_sentences(text_with_pos, word_to_index_dic_with_pos)
ner_with_index = get_index_list_of_ner(text, ner_to_index_dic)

# 7. 모델 용 input data 만들기
X_input = [padding(index) for index in sentences_with_index]
X_input_with_pos = [padding(index) for index in sentences_with_index_with_pos]
y_input = [padding(index) for index in ner_with_index]

# 8. 모델 용 test input data 만들기
# test['word'] = test['word'].str.lower()

test_text = get_text_list(test, 'sentence_index', pos_tag=False)
test_text_with_pos = get_text_list(test, 'sentence_index')

test_sentences_with_index = get_index_list_of_sentences(
    test_text, word_to_index_dic, pos_tag=False)
test_sentences_with_index_with_pos = get_index_list_of_sentences(
    test_text_with_pos, word_to_index_dic_with_pos)

test_ner_with_index = get_index_list_of_ner(test_text, ner_to_index_dic)
X_test = [padding(index) for index in test_sentences_with_index]
X_test_with_pos = [padding(index) for index in test_sentences_with_index_with_pos]
y_test = [padding(index) for index in test_ner_with_index]

# 파일로 저장하기
np.save('./dataset/train_input', X_input)
np.save('./dataset/train_input_with_pos', X_input_with_pos)
np.save('./dataset/train_labels', y_input)

np.save('./dataset/test_input', X_test)
np.save('./dataset/test_input_with_pos', X_test_with_pos)
np.save('./dataset/test_labels', y_test)

# 단어 관련된 정보 저장하기

WORDS_CONFIG = {
    'word_to_index_dic': word_to_index_dic,
    'index_to_word_dic': index_to_word_dic,
    'index_to_word_dic_with_pos': index_to_word_dic_with_pos,
    'ner_to_index_dic': ner_to_index_dic,
    'index_to_ner_dic': index_to_ner_dic,
    'vocab_size': len(word_to_index_dic),
    'vocab_size_with_pos': len(word_to_index_dic_with_pos),
    'n_labels': len(ner_to_index_dic),
}

json_dumps = json.dumps(WORDS_CONFIG)
with open('./CONFIG/WORDS_CONFIG.json', 'w') as f:
    f.write(json_dumps)
