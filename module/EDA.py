import os
import re
import csv


def preview_text_file(DATA_PATH, n_line=5):
    with open(DATA_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines[:n_line]:
            print(line)


def get_test_file_as_list(DATA_PATH):
    with open(DATA_PATH, 'r') as f:
        lines = f.readlines()
    return lines


def get_clean_text(text_raw):
    text_clean = []
    sentence = []
    for line in text_raw:
        if line.startswith('-DOCSTART') or line.startswith('\n'):
            if sentence:
                text_clean.append(sentence)
                sentence = []
            continue
        line = re.sub(r'\n', '', line)
        line_list = line.split()
        sentence.append(line_list)
    return text_clean


def get_word_count_dic(text_clean):
    count_dic = {}
    for sentence in text_clean:
        for content in sentence:
            word = content[0]
            if word not in count_dic.keys():
                count_dic[word] = 1
            else:
                count_dic[word] += 1
    return count_dic


def get_ner_count_dic(text_clean):
    count_dic = {}
    for sentence in text_clean:
        for content in sentence:
            ner = content[3]
            if ner not in count_dic.keys():
                count_dic[ner] = 1
            else:
                count_dic[ner] += 1
    return count_dic


def save_csv_file(save_root, file_name, text):
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    with open(os.path.join(save_root, file_name), 'w') as csv_write:
        writer = csv.writer(csv_write)
        writer.writerow(['sentence_index', 'word', 'pos', 'chunk', 'ner'])
        for i, sentence in enumerate(text):
            for content in sentence:
                writer.writerow(content)
