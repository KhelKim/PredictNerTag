import os
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras_contrib.layers import CRF

import warnings
warnings.filterwarnings("ignore")

# 파일 불러오기
DATA_ROOT = 'dataset'
TRAIN_INPUT = 'train_input.npy'
TRAIN_LABELS = 'train_labels.npy'

CONFIG_ROOT = 'CONFIG'
CONFIG_POS_TAG = 'POS_TAG_CONFIG.json'
CONFIG_WORDS = 'WORDS_CONFIG.json'

TRAIN_INPUT_PATH = os.path.join(DATA_ROOT, TRAIN_INPUT)
TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, TRAIN_LABELS)

POS_TAG_PATH = os.path.join(CONFIG_ROOT, CONFIG_POS_TAG)
WORDS_PATH = os.path.join(CONFIG_ROOT, CONFIG_WORDS)

train_input = np.load(TRAIN_INPUT_PATH)
train_labels = np.load(TRAIN_LABELS_PATH)

with open(POS_TAG_PATH) as pos_tag_path:
    pos_tag_config = json.load(pos_tag_path)

with open(WORDS_PATH) as words_path:
    words_config = json.load(words_path)

max_sentence_length = pos_tag_config['max_sentence_length']
vocab_size = words_config['vocab_size']
ner_to_index_dic = words_config['ner_to_index_dic']
n_labels = len(ner_to_index_dic)

# 테스트와 검증 셋 나누기

X_train, X_dev, y_train, y_dev = train_test_split(
    train_input, train_labels, test_size=0.1, random_state=42)

# label 원핫 벡터로 바꾸기
train_labels_one_hot = np_utils.to_categorical(train_labels)

# hyper parameter ####################################################################
embedding_size = pos_tag_config['embedding_size']
n_hidden1 = pos_tag_config['n_hidden1']
n_hidden2 = pos_tag_config['n_hidden2']
batch_size = pos_tag_config['batch_size']
epochs = pos_tag_config['epochs']
######################################################################################

# 모델 구조
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size,
                    input_length=max_sentence_length, mask_zero=True))
model.add(Bidirectional(
    LSTM(n_hidden1, return_sequences=True, recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(n_hidden2, activation='relu')))
crf = CRF(n_labels)
model.add(crf)

model.compile(loss=crf.loss_function, optimizer=Adam(
    0.001), metrics=[crf.accuracy])

# 훈련
history = model.fit(train_input, train_labels_one_hot, batch_size=batch_size,
                    epochs=epochs, validation_split=0.1, verbose=1)

# 모델 저장하기
model_json = model.to_json()
with open("./model/model.json", 'w') as json_file:
    json_file.write(model_json)

model.save_weights("./model/model.h5")

json_dumps = json.dumps(history.history)
with open('./model/model_history.json', 'w') as f:
    f.write(json_dumps)

print("keras model is saved")
