import re
import numpy as np
import tensorflow as tf
import string


punct_marks = tuple('.,!?;"\'')
english = tuple(['ENG']) + tuple(string.ascii_lowercase)
digits = tuple(['DIG']) + tuple(string.digits)
russian = tuple('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
real_alphabet = russian + tuple(' -.*')
alphabet = real_alphabet + tuple(['NULL', 'DIG', 'ENG'])

alpha2idx = {c: idx for idx, c in enumerate(alphabet)}
null_idx = alpha2idx['NULL']


reduction = {c: c for c in alphabet}
for lst in punct_marks, english, digits:
    for c in lst:
        reduction[c] = lst[0]
        if c.lower():
            reduction[c.upper()] = lst[0]
for c in russian:
    reduction[c.upper()] = c


def enc(c):
    return alpha2idx[reduction.get(c, '*')]


Y_NULL = 2


def encode_text(text):
    encoding = []
    for c in text:
        encoding.append(enc(c))
    return np.array(encoding)


def encode_batch(batch, with_y=True):

    def first_or_whole(x):
        if isinstance(x, (list, tuple)):
            return x[0]
        return x
    
    max_len = max(len(first_or_whole(sample)) for sample in batch)
    X = np.full([len(batch), max_len], null_idx)
    if with_y:
        y = np.full_like(X, Y_NULL)
    for idx, sample in enumerate(batch):
        sample_len = len(first_or_whole(sample))
        X[idx, :sample_len] = encode_text(first_or_whole(sample))
        if with_y:
            y[idx, :sample_len] = sample[1]
    if with_y:
        return X, y
    return X


THRESHOLD = 0.75


def mark_text(graph, model, text):
    def is_approx_vulg(cls):
        window = 3
        if len(cls) < window and cls.sum() == len(cls):
            return True
        for idx in range(len(cls) - window + 1):
            if cls[idx:idx+window].sum() == window:
                return True
        return False

    if not text:
        return ''
    batch = encode_batch([text], with_y=False)
    with graph.as_default():
        [pred] = model.predict(batch)
    cls = (pred >= THRESHOLD).astype(np.int)
    for occ in re.finditer(r'\w+', text):
        left, right = occ.span()
        if is_approx_vulg(cls[left:right]):
            cls[left:right] = 1
        else:
            cls[left:right] = 0
    new_text = []
    for c, cl in zip(text, cls):
        new_text.append([c, '<span style="color:red">' + c + '</span>'][cl])
    return ''.join(new_text)
