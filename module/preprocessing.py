def get_text_list(data, index_name, pos_tag=True):
    text = []
    for cur_sentence in data.groupby(index_name):
        sentence = []
        for content in cur_sentence[1].values:
            sentence_id, word, pos, chunk, ner = content
            if pos_tag:
                sentence.append([(word, pos), ner])
            else:
                sentence.append([word, ner])
        text.append(sentence)
    return text


def get_word_to_index_dic(text, word_count_dic, min_word_count, pos_tag=True):
    word_to_index_dic = {
        "PAD": 0,
    }
    i = 1
    for sentence in text:
        for word, ner in sentence:
            if word_count_dic[word] > min_word_count:
                if word not in word_to_index_dic.keys():
                    word_to_index_dic[word] = i
                    i += 1
            else:
                if pos_tag:
                    pos = word[1][:2]
                    if ('OOV', pos) not in word_to_index_dic.keys():
                        word_to_index_dic[('OOV', pos)] = i
                        i += 1
                else:
                    if 'OOV' not in word_to_index_dic.keys():
                        word_to_index_dic['OOV'] = i
                        i += 1
    return word_to_index_dic


def get_ner_to_index_dic(text):
    ner_to_index_dic = {
        'PAD': 0,
    }
    j = 1
    for sentence in text:
        for word, ner in sentence:
            if ner not in ner_to_index_dic.keys():
                ner_to_index_dic[ner] = j
                j += 1
    return ner_to_index_dic


def get_index_list_of_sentences(text, word_to_index_dic, pos_tag=True):
    sentences_with_index = []
    for sentence in text:
        word_index = []
        for word, ner in sentence:
            if word in word_to_index_dic.keys():
                word_index.append(word_to_index_dic[word])
            else:
                if pos_tag:
                    if ("OOV", word[1]) in word_to_index_dic.keys():
                        word_index.append(word_to_index_dic[("OOV", word[1])])
                    else:
                        word_index.append(word_to_index_dic[("OOV", "NN")])
                else:
                    word_index.append(word_to_index_dic["OOV"])
        sentences_with_index.append(word_index)
    return sentences_with_index


def get_index_list_of_ner(text, ner_to_index_dic):
    ner_with_index = []
    for sentence in text:
        ner_index = []
        for word, ner in sentence:
            ner_index.append(ner_to_index_dic[ner])
        ner_with_index.append(ner_index)
    return ner_with_index
