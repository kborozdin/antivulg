from subprocess import Popen, PIPE
import pickle
import re
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
import threading

from new_model import *


lock = threading.Lock()
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

sentences = []
in_process = set()


def initialize():
    old_model = load_model('model')
    graph = tf.get_default_graph()
    proc = Popen('../fastText/fasttext print-word-vectors ../cc.ru.300.bin',
                 stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding='utf-8', shell=True)
    app = Flask(__name__)

    ### Start new model defs ###
    def custom_loss_function(y_true, y_pred):
        EPS = 1e-10
        weight = tf.cast(tf.not_equal(y_true, Y_NULL), dtype=tf.float32)
        loss = -weight * (y_true * tf.log(y_pred + EPS) + (1 - y_true) * tf.log(1 - y_pred + EPS))
        return tf.reduce_mean(loss)

    model = load_model('model_final.h5',
                       custom_objects={'tf': tf,
                                       'custom_loss_function': custom_loss_function,
                                       'null_idx': null_idx})
    ### End new model defs ###

    return app, old_model, graph, proc, model


app, old_model, graph, proc, model = initialize()


def tokenize(text):
    return [word for word in re.split(r'[\s.,;:?!]+', text) if word]


def convert_to_vectors(words):
    VEC_SIZE = 300
    data = np.zeros((len(words), VEC_SIZE))
    for i, word in enumerate(words):
        proc.stdin.write(word.lower() + '\n')
        proc.stdin.flush()
        vec = proc.stdout.readline()
        data[i, :] = np.array(vec.split()[-VEC_SIZE:], dtype=np.float64)
    return data


def old_mark_text(text):
    words = tokenize(text)
    if not words:
        words.append('пустой_запрос')
    with open('log.txt', 'a') as f:
        for word in words:
            f.write(word + '\n')
    data = convert_to_vectors(words)
    with graph.as_default():
        predictions = old_model.predict_classes(data)
    output = []
    for word, label in zip(words, predictions):
        if label == 0 or len(word) < 3:
            output.append(word)
        else:
            output.append('<span style="color:red">' + word + '</span>')
    return ' '.join(output)


@app.before_first_request
def before_first_request():
    global sentences
    with open(os.path.join(APP_ROOT, '../data/unlabeled10k.txt'), 'r', encoding='utf=8') as f:
        sentences = set([line.strip() for line in f])
    try:
        with open(os.path.join(APP_ROOT, 'processed.txt'), 'r', encoding='utf=8') as f:
            processed = set([line.strip() for line in f])
    except:
        with open(os.path.join(APP_ROOT, 'processed.txt'), 'w+', encoding='utf=8') as f:
            processed = set([line.strip() for line in f])
    sentences = list(sentences - processed)


@app.route('/old', methods=['POST'])
def old_app_post():
    query = request.form["query"]
    return render_template('old_app.html', query=query, response=old_mark_text(query))


@app.route('/old', methods=['GET'])
def old_app_get():
    return render_template('old_app.html', query='')


@app.route('/', methods=['POST'])
def app_post():
    query = request.form["query"]
    return render_template('app.html', query=query, response=mark_text(graph, model, query))


@app.route('/', methods=['GET'])
def app_get():
    return render_template('app.html', query='')


@app.route('/marking', methods=['GET', 'POST'])
def marking():
    global sentences
    global in_process
    if request.method == 'GET':
        with lock:
            if len(sentences) == 0:
                return render_template('ending title.html')
            if request.is_xhr:
                sentence = sentences.pop()
                in_process.add(sentence)
                return jsonify({'text': sentence})
            else:
                return render_template('marking.html')
    if request.method == 'POST':
        sentence = request.get_json()['text']
        result_arr = request.get_json()['resultArr']
        with lock:
            if sentence in in_process:
                in_process.remove(sentence)
                with open(os.path.join(APP_ROOT, 'labeled.txt'), 'a', encoding='utf=8') as f:
                    f.write("{0}\n{1}\n".format(sentence, "".join(map(str, result_arr))))
                with open(os.path.join(APP_ROOT, 'processed.txt'), 'a', encoding='utf=8') as f:
                    f.write("{0}\n".format(sentence))
            if len(sentences) == 0:
                return jsonify({'redirect': ''})
            sentence = sentences.pop()
            in_process.add(sentence)
            return jsonify({'text': sentence})
