
import tensorflow as tf
from tensorflow import Tensor
from .image_generator import ImageGenerator



class Inputs(object):
    def __init__(self, img):
        self.img = img


class Dataset(object):

    def __init__(self, generator = ImageGenerator(images_path = '/Users/pablomartin/python/Mushroom_Classifier/flat_images'),
                      batch_size = 10,
                      prefetch_batch_buffer = 5,
                      target_size = 299,
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

    def add_jpeg_decoding(module_spec):
      """Adds operations that perform JPEG decoding and resizing to the graph..

      Args:
        module_spec: The hub.ModuleSpec for the image module being used.

      Returns:
        Tensors for the node to feed JPEG data into, and the output of the
          preprocessing steps.
      """
      input_height, input_width = hub.get_expected_image_size(module_spec)
      input_depth = hub.get_num_image_channels(module_spec)
      jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
      decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
      # Convert from full range of uint8 to range [0,1] of float32.
      decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                            tf.float32)
      decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
      resize_shape = tf.stack([input_height, input_width])
      resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
      resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                               resize_shape_as_int)
      return jpeg_data, resized_image
