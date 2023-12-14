import pandas as pd
import numpy as np
from collections import Counter
import numpy as np
import re

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