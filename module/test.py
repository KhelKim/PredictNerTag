import numpy as np


def get_arg_list(values):
    result = []
    for sentence in values:
        for word in sentence:
            ner_index = np.argmax(word)
            result.append(ner_index)
    return result


def get_f1_score(confusion_matrix, B_PER_index, I_PER_index):
    TP_FP = np.sum(confusion_matrix[:, [B_PER_index, I_PER_index]])
    TP_FN = np.sum(confusion_matrix[[B_PER_index, I_PER_index], :])
    TP = np.sum(confusion_matrix[[B_PER_index, I_PER_index],
                                 [B_PER_index, I_PER_index]])

    precision = TP / TP_FP
    recall = TP / TP_FN
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score


def get_accuracy(test_predict, test_labels, B_PER_index, I_PER_index):
    test_predict_arg_by_sentence = []
    for sentence in test_predict:
        temp = []
        for ner in sentence:
            temp.append(np.argmax(ner))
        test_predict_arg_by_sentence.append(temp)

    count = 0
    for pred, true in zip(test_predict_arg_by_sentence, test_labels):
        if (B_PER_index in pred and B_PER_index in true) or (I_PER_index in pred and I_PER_index in true):
            count += 1
        elif (B_PER_index not in pred and B_PER_index not in true) or (I_PER_index not in pred and I_PER_index not in true):
            count += 1
    accuracy = count / len(test_labels)
    return accuracy
