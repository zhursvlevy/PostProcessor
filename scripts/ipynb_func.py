import pandas as pd
import numpy as np
from collections import Counter
import numpy as np
import re
import matplotlib.pyplot as plt

def merge_dataset(roots: list) -> pd.DataFrame:
    """ 
    Function to merge all files, whose paths are in list "roots".
    roots -- list of roots to dataset files in .parquet extension
    """
    data = []
    for root in roots:
        if isinstance(data, pd.DataFrame):
            data = pd.concat([data, pd.read_parquet(root)])
        else:
            data = pd.read_parquet(root)
    return data


def getwordlist(tags: pd.Series) -> list:
    """ 
    Returns list of all words, included in the transmitted pandas.Series.
    Data repetitions and order are preserved.
    """
    tags_list = []
    #[[tags_list.append(tag) for tag in tags.tolist()] for tags in data.tags]
    [tags_list.extend(tag) for tag in tags.tolist()]
    return tags_list


def getworddict(tags_list: list, at_least=1, sort=True, reverse=True) -> dict:
    """ 
    Returns dict of counted words in the transmitted list. 
    No duplicate data, order controls with parameters "sort" and "reverse".

    If "sort" == True, dict sorts in order, which determines by "reverse."
        If "reverse" == True, then dict sorts in descending order. Else in ascending order.
    """
    assert at_least >= 1, "Minimum number of tags must be more or equal to 1."
    tgdict = Counter(tags_list)
    dct = {k:tgdict[k] for k in tgdict if tgdict[k] >= at_least}
    if sort:
        return dict(sorted(dct.items(), key=lambda x: x[1], reverse=reverse))
    return dct


def removePostsByTags(data, badtags: list):
    tag_mask = np.sum([[btag in tag for btag in badtags] for tag in data.tags], axis=1).tolist()
    tag_mask = list(map(bool, tag_mask))
    tag_mask = [not elem for elem in tag_mask]
    return data[tag_mask]


def removeTags(formated_tags, permitted_tags: list, prohibitted_tags: list):
    for i in range(len(formated_tags)):
        for tag_num in range(len(formated_tags[i])):
            if formated_tags[i][tag_num] not in permitted_tags or formated_tags[i][tag_num] in prohibitted_tags:
                formated_tags[i][tag_num] = ''

    for i in range(len(formated_tags)):
        new = []
        for j in range(len(formated_tags[i])):
            if formated_tags[i][j] != '':
                new.append(formated_tags[i][j])
        formated_tags[i] = new 
    
    return formated_tags

def removeEmptyPosts(tags):
    mask = []
    for i in range(len(tags)):
        if len(tags[i]) == 0:
            mask.append(False)
        else:
            mask.append(True)
    return mask

def formateTags(data):
    tags = [tag_line for tag_line in data.tags]
    tags = [re.split(r';', tags) for tags in tags]
    return tags

def lower_tags(df):
    tmp = df.tags
    if isinstance(df.iloc[0]['tags'], np.ndarray):
        tmp = []
        for tgs in df.tags:
            tmp.append(tgs.tolist())
    lowertgs = []
    for tgs in tmp:
        lowertgs.append(list(map(lambda x: x.lower(), tgs)))
    df.tags = lowertgs

def recallk(df_true_tags, df_pred_tags, k=None):
    #count = 0
    #total_count = 0
    #for true_tags, pred_tags in zip(df_true_tags, df_pred_tags):
    #    total_count += 1
    #    if k is not None:
    #        for pred_tag in pred_tags[:k]:
    #            if pred_tag in true_tags:
    #                count += 1
    #                continue
    #    else:
    #        for pred_tag in pred_tags:
    #            if pred_tag in true_tags:
    #                count += 1
    #                continue
    #return count/total_count
    recalls = []
    for pred, true in zip(df_pred_tags, df_true_tags):
        pred = pred[:k]
        numenator = len(set(pred).intersection(true))
        denometator = len(true)
        recalls.append(numenator/denometator)
    return np.mean(recalls), np.median(recalls)


def get_y_pred_items(model_out, out_Vectorizer, THR):
    y_pred_items = []
    for lbls in model_out:
        tmp = []
        for i in range(len(lbls)):
            if abs(lbls[i]) < THR:
                tmp.append((out_Vectorizer.get_feature_names_out()[i], lbls[i]))
        y_pred_items.append(tmp)
    sorted_y_pred_items = [sorted(elems, key=lambda x: x[1], reverse=True) for elems in y_pred_items]
    return sorted_y_pred_items

def get_labels(model_out, out_Vectorizer, THR):
    sorted_y_pred = get_y_pred_items(model_out, out_Vectorizer, THR)
    pred_labels = [[elem[0] for elem in elems] for elems in sorted_y_pred]
    return pred_labels

def plot_model_history(history, title='Model loss and acc during training'):
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    val_acc = history['val_acc']

    fig, axs = plt.subplot_mosaic([[1, 2], [1, 3]], figsize=(17, 8))
    fig.suptitle(title)
    fig.subplots_adjust(top=0.92)

    axs[1].plot([i for i in range(len(train_loss))], train_loss)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Loss during training of a model')
    axs[1].grid()


    axs[2].plot([i+1 for i in range(len(val_loss))], val_loss)
    axs[2].set_xticks([i+1 for i in range(len(val_acc))])
    axs[2].set_ylabel('Loss')
    axs[2].set_title('Loss on validation')
    axs[2].grid()

    axs[3].plot([i+1 for i in range(len(val_acc))], val_acc)
    axs[3].set_xticks([i+1 for i in range(len(val_acc))])
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Accuracy')
    axs[3].set_title('Accuracy on validation')
    axs[3].grid()


def sorted_labels(labels, probas, Vectorizer):
    val_prob = []
    for i in range(len(probas)):
        tmp = []
        for j in range(len(probas[i])):
            if labels.toarray()[i][j] != 0:
                tmp.append(probas[i][j])
        val_prob.append(tmp)

    sort_y = []
    for tags, probs in zip(Vectorizer.inverse_transform(labels), val_prob):
        tmp = []
        for tag, prob in zip(tags, probs):
            tmp.append((tag, prob))
        sort_y.append(tmp)

    sort_y = [sorted(tags, key=lambda x: -x[1]) for tags in sort_y]

    sss = []
    for tags in sort_y:
        tmp = []
        for tag in tags:
            tmp.append(tag[0])
        sss.append(tmp)
    return sss