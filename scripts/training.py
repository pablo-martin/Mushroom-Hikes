from image_generator import ImageGenerator
from tf_dataset import Inputs, Dataset

if __init__ == '__main__':
    generator = ImageGenerator(images_path = '/Users/pablomartin/python/Mushroom_Classifier/flat_images')
    iter = generator.get_next_image_random()
    for i in range(2):
        print(next(iter))
    ds = Dataset(generator)
    model_input = ds.next_element
    module_spec = hub.load_module_spec("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
    module = hub.Module(module_spec)


    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        images = sess.run(model_input.img)
        features = module(images)
        image_feature_vecs = sess.run(features)
