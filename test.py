import os
import json
from keras.models import model_from_json
from keras.optimizers import Adam
from keras_contrib.layers import CRF
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from module.test import *

import warnings
warnings.filterwarnings("ignore")

# 파일 불러오기
DATA_ROOT = 'dataset'
TEST_INPUT = 'test_input.npy'
TEST_LABELS = 'test_labels.npy'

CONFIG_ROOT = 'CONFIG'
CONFIG_POS_TAG = 'POS_TAG_CONFIG.json'
CONFIG_WORDS = 'WORDS_CONFIG.json'
POS_TAG_PATH = os.path.join(CONFIG_ROOT, CONFIG_POS_TAG)
WORDS_PATH = os.path.join(CONFIG_ROOT, CONFIG_WORDS)

MODEL_ROOT = 'model'
MODEL_JSON = 'model.json'
MODEL_H5 = 'model.h5'

TEST_INPUT_PATH = os.path.join(DATA_ROOT, TEST_INPUT)
TEST_LABELS_PATH = os.path.join(DATA_ROOT, TEST_LABELS)

with open(POS_TAG_PATH) as pos_tag_path:
    pos_tag_config = json.load(pos_tag_path)

with open(WORDS_PATH) as words_path:
    words_config = json.load(words_path)

vocab_size = words_config['vocab_size']
word_to_index_dic = words_config['word_to_index_dic']
ner_to_index_dic = words_config['ner_to_index_dic']
B_PER_index = ner_to_index_dic['B-PER']
I_PER_index = ner_to_index_dic['I-PER']
n_labels = len(ner_to_index_dic)

MODEL_JSON_PATH = os.path.join(MODEL_ROOT, MODEL_JSON)
MODEL_H5_PATH = os.path.join(MODEL_ROOT, MODEL_H5)

test_input = np.load(TEST_INPUT_PATH)
test_labels = np.load(TEST_LABELS_PATH)
test_labels_one_hot = np_utils.to_categorical(test_labels)

with open(MODEL_JSON_PATH, 'r') as f:
    loaded_model_json = f.read()

loaded_model = model_from_json(loaded_model_json, custom_objects={'CRF': CRF})
loaded_model.load_weights(MODEL_H5_PATH)

crf = CRF(n_labels)

loaded_model.compile(loss=crf.loss_function,
                     optimizer=Adam(0.001), metrics=[crf.accuracy])

test_predict = loaded_model.predict(test_input)
test_predict_arg = get_arg_list(test_predict)

confusion_matrix = confusion_matrix(
    y_true=test_labels.flat, y_pred=test_predict_arg)

accuracy = get_accuracy(test_predict, test_labels, B_PER_index, I_PER_index)
f1_score = get_f1_score(confusion_matrix, B_PER_index, I_PER_index)


# 출력
division_length = 80
print("confusion matrix  ".ljust(division_length, '-'))
print('B PER index, I PER index:', B_PER_index, I_PER_index)
for line in confusion_matrix:
    for number in line:
        print(str(number).ljust(10, ' '), end='')
    print()
print("-" * division_length)
print()
print()

print("accuracy  ".ljust(division_length, '-'))
print(accuracy)
print("-" * division_length)
print()
print()

print("f1 score  ".ljust(division_length, '-'))
print(f1_score)
print("-" * division_length)
print()
print()

with open("result/result.txt", 'a') as f:
    f.write(f'Without pos tag\n')
    f.write(f'CONFIG: {pos_tag_config}\n')
    f.write(f"accuracy: {accuracy}\nf1_score: {f1_score}\n\n")
