import tensorflow as tf
from scripts.image_generator import ImageSplitter, ImageGenerator, Dataset

Image_Lists = ImageSplitter('bottleneck_files/',10,10,150)
training_generator = ImageGenerator(Image_Lists, 'training', 0)
D = Dataset(training_generator)
number_testing_images = \
                sum([len(Image_Lists.image_lists[w]['testing']) \
                for w in Image_Lists.image_lists])

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

graph = load_graph('front_end/graph/one_layer_full_data.pb')
graph = load_graph('output_graph.pb')

input_operation = graph.get_operation_by_name('import/input/BottleneckInputPlaceholder')
output_operation = graph.get_operation_by_name('import/final_result')

top_5_accuracy = []
K = 10
for no_steps in range(80):
    print('on iteration %i' %no_steps)
    with tf.Session() as sess:
        bottlenecks, ground_truth, _ = sess.run(D.next_element)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0],
            {input_operation.outputs[0] : bottlenecks})


    for vec, gt in zip(results, ground_truth):
        top_5 = [np.where(vec==w)[0][0] for w in sorted(vec)[-K:]]
        top_5_accuracy.append(int(gt in top_5))
total_accuracy = np.mean(top_5_accuracy)
print('top %i accuracy: %1.2f%%' %(K, total_accuracy * 100))
