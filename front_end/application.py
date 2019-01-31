from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from flask import Flask, render_template, flash, redirect, url_for, request
from config import Config
from flask_bootstrap import Bootstrap

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

BASE_DIR = os.environ['HOME'] + '/python/Mushroom_Classifier/'
model_file = BASE_DIR + 'front_end/graph/one_layer_full_data.pb'
label_file = BASE_DIR + 'front_end/graph/one_layer_full_data_labels.txt'

module_spec = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"
module_spec = hub.load_module_spec(module_spec)
input_height, input_width = hub.get_expected_image_size(module_spec)
input_mean = 0
input_std = 255
input_name = 'import/Placeholder'
output_name = "import/final_result"

application = app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def read_tensor_from_image_file(file_name,
                            input_height=299,
                            input_width=299,
                            input_mean=0,
                            input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label




#load model and labels when you start application
graph = load_graph(model_file)
labels = load_labels(label_file)


def classify(file_name, top_k = 5):
    t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)


    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: t})
    results = np.squeeze(results)

    top_k = {labels[w]:results[w] for w in results.argsort()[-top_k:][::-1]}
    top_k = sorted(top_k.items(), key=lambda x: x[1], reverse=True)

    id_message = ''
    for el in top_k:
        id_message += 'Species: %s <br/>Probability: %1.2f%%<br/><br/>' \
                                                    %(el[0], el[1] * 100)
    flash(id_message)
    return

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', title='Home')


@app.route('/Upload')
def Upload():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file_name = os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg')
        file.save(file_name)
        classify(file_name)
    return redirect(url_for('Upload'))


if __name__ == '__main__':
    print('sensible eyes')
    application.run(debug=True)
