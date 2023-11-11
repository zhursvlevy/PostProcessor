import pandas as pd
from collections import Counter

def merge_dataset(roots: list) -> pd.core.frame.DataFrame:
    """ 
    Function to merge all files, whose paths are in list "roots".
    roots -- list of roots to dataset files in .parquet extension
    """
    data = []
    for root in roots:
        if isinstance(data, pd.core.frame.DataFrame):
            data = pd.concat([data, pd.read_parquet(root)])
        else:
            data = pd.read_parquet(root)
    return data

def get_tags(tags: pd.core.series.Series) -> list:
    """ 
    Returns list of all words, included in the transmitted pandas.Series.
    Data repetitions and order are preserved.
    """
    tags_list = []
    #[[tags_list.append(tag) for tag in tags.tolist()] for tags in data.tags]
    [tags_list.extend(tag) for tag in tags.tolist()]
    return tags_list

def tags_counter(tags_list: list, at_least=1, sort=True, reverse=True) -> dict:
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