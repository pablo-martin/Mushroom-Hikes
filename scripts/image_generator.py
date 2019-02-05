import os
import re
import hashlib
import glob
import random
import collections
import tensorflow as tf
from tensorflow import Tensor

class Inputs(object):
	def __init__(self, bottlenecks, species_ground_truths, filenames):
		self.bottlenecks = bottlenecks
		self.species_ground_truths = species_ground_truths
		self.filenames = filenames

class ImageSplitter(object):
	def __init__(self,
				image_dir,
				testing_percentage,
				validation_percentage,
				LONG_TAIL_CUTOFF):

		self.image_dir = image_dir
		self.testing_percentage = testing_percentage
		self.validation_percentage = validation_percentage
		self.LONG_TAIL_CUTOFF = LONG_TAIL_CUTOFF
		self.image_lists = self.create_image_lists()
		self.species_classes = len(self.image_lists.keys())
		self.genus_classes = len(set([self.image_lists[w]['dir'].split('_')[0] \
										for w in self.image_lists.keys()]))

	def create_image_lists(self):
		"""Builds a list of training images from the file system.

		Analyzes the sub folders in the image directory, splits them into stable
		training, testing, and validation sets, and returns a data structure
		describing the lists of images for each label and their paths.

		This function is adapted from Tensorflow:
		curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py
		"""
		MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1
		if not tf.gfile.Exists(self.image_dir):
			tf.logging.error("Image directory '" + self.image_dir + "' not found.")
			return None
		result = collections.OrderedDict()
		sub_dirs = sorted(x[0] for x in tf.gfile.Walk(self.image_dir))
		# The root directory comes first, so skip it.
		is_root_dir = True
		for sub_dir in sub_dirs:
			if is_root_dir:
				is_root_dir = False
				continue
			extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
									for ext in ['JPEG', 'JPG', 'jpeg', 'jpg', 'txt']))
			file_list = []
			dir_name = os.path.basename(sub_dir)
			if dir_name == self.image_dir:
				continue
			tf.logging.info("Looking for images in '" + dir_name + "'")
			for extension in extensions:
				file_glob = os.path.join(self.image_dir, dir_name, '*.' + extension)
				file_list.extend(tf.gfile.Glob(file_glob))
			if not file_list:
				tf.logging.warning('No files found')
				continue
			if len(file_list) < self.LONG_TAIL_CUTOFF:
				tf.logging.warning(
				  'WARNING: Folder has less than {} images, which may cause issues.'.format(len(file_list)))
				long_tail = False
			else:
				long_tail = True

			label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
			training_images = []
			testing_images = []
			validation_images = []
			for file_name in file_list:
				base_name = os.path.basename(file_name)
				# We want to ignore anything after '_nohash_' in the file name when
				# deciding which set to put an image in, the data set creator has a way of
				# grouping photos that are close variations of each other. For example
				# this is used in the plant disease data set to group multiple pictures of
				# the same leaf.
				hash_name = re.sub(r'_nohash_.*$', '', file_name)
				# This looks a bit magical, but we need to decide whether this file should
				# go into the training, testing, or validation sets, and we want to keep
				# existing files in the same set even if more files are subsequently
				# added.
				# To do that, we need a stable way of deciding based on just the file name
				# itself, so we do a hash of that and then use that to generate a
				# probability value that we use to assign it.
				hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
				percentage_hash = ((int(hash_name_hashed, 16) %
								  (MAX_NUM_IMAGES_PER_CLASS + 1)) *
								 (100.0 / MAX_NUM_IMAGES_PER_CLASS))
				if percentage_hash < self.validation_percentage:
					validation_images.append(base_name)
				elif percentage_hash < (self.testing_percentage + self.validation_percentage):
					testing_images.append(base_name)
				else:
					training_images.append(base_name)
			result[label_name] = {
				'dir': dir_name,
				'long_tail' : long_tail,
				'training': training_images,
				'testing': testing_images,
				'validation': validation_images,
			}
		return result

class ImageGenerator(object):
	def __init__(self,
				imagesplitter : ImageSplitter,
				category,
				balanced):

		self.image_lists = imagesplitter.image_lists
		self.image_dir = imagesplitter.image_dir
		self.category = category
		self.balanced = balanced
		self.long_tail_species = \
			[w for w in self.image_lists if self.image_lists[w]['long_tail']]
		self.short_tail_species = \
			[w for w in self.image_lists if not self.image_lists[w]['long_tail']]
		self.long = True
		self.Genus = self.build_genus()

	def build_genus(self):
		return list(set([self.image_lists[w]['dir'].split('_')[0] \
											for w in self.image_lists.keys()]))

	def get_next_image(self):
		while True:
			if not self.balanced:
				# draw a species index at random
				species_index = random.randrange(len(self.image_lists.keys()))
				species_label = list(self.image_lists.keys())[species_index]
			else:
				if self.long:
					species_label = random.choice(self.long_tail_species)
					self.long = False
				else:
					species_label = random.choice(self.short_tail_species)
					self.long = True
				species_index = list(self.image_lists.keys()).index(species_label)

			genus_label = self.image_lists[species_label]['dir'].split('_')[0]
			genus_index = self.Genus.index(genus_label)
			# draw a particular photo at random from right category
			ex = random.choice(self.image_lists[species_label][self.category])

			#get path and read image embedding "bottleneck"
			bottleneck_path = os.path.join(self.image_dir, self.image_lists[species_label]['dir'], ex)

			with open(bottleneck_path, 'r') as bottleneck_file:
				bottleneck_string = bottleneck_file.read()
			try:
				bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
			except ValueError:
				tf.logging.warning('Invalid float found, recreating bottleneck')
			yield (bottleneck_values, genus_index, species_index, bottleneck_path)


class Dataset(object):
	def __init__(self, generator : ImageGenerator,
					  batch_size = 100,
					  prefetch_batch_buffer = 50,
					  balanced = 0):
		self.batch_size = batch_size
		self.prefetch_batch_buffer = prefetch_batch_buffer
		self.next_element = self.build_iterator(generator.get_next_image)

	def build_iterator(self, generator_func):

		dataset = tf.data.Dataset.from_generator(generator_func,
								output_types=(tf.float32, tf.int64, tf.int64, tf.string))
		dataset = dataset.batch(self.batch_size)
		dataset = dataset.prefetch(self.prefetch_batch_buffer)
		iter = dataset.make_one_shot_iterator()
		return iter.get_next()
