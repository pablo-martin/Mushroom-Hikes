
import tensorflow as tf
from tensorflow import Tensor
from .image_generator import ImageGenerator



class Inputs(object):
    def __init__(self, img):
        self.img = img


class Dataset(object):
    img1_resized = 'img1_resized'
    img2_resized = 'img2_resized'
    label = 'same_person'

    def __init__(self, generator = ImageGenerator(images_path = '/Users/pablomartin/python/Mushroom_Classifier/flat_images'),
                      batch_size = 10,
                      prefetch_batch_buffer = 5,
                      target_size = 128,
                      balanced = 0):
        self.batch_size = batch_size
        self.prefetch_batch_buffer = prefetch_batch_buffer
        self.target_size = target_size
        if not balanced:
            self.next_element = self.build_iterator(generator.get_next_image_random)
        else:
            self.next_element = self.build_iterator(generator.get_next_image_balanced)

    def build_iterator(self, generator_func):

        dataset = tf.data.Dataset.from_generator(generator_func,
                                                 output_types=(tf.string))
        dataset = dataset.map(self._read_image_and_resize)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_batch_buffer)
        iter = dataset.make_one_shot_iterator()
        element = iter.get_next()
        return Inputs(element)


    def _read_image_and_resize(self, image_gen):
        target_size = [self.target_size, self.target_size]
        # read images from disk
        img_file = tf.read_file(image_gen)
        img1 = tf.image.decode_image(img_file)

        # let tensorflow know that the loaded images have unknown dimensions, and 3 color channels (rgb)
        img1.set_shape([None, None, 3])

        return tf.image.resize_images(img1, target_size)
