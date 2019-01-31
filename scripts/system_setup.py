import re
import hashlib
import collections

def create_image_lists(image_dir, testing_percentage, validation_percentage):
	"""Builds a list of training images from the file system.

	Analyzes the sub folders in the image directory, splits them into stable
	training, testing, and validation sets, and returns a data structure
	describing the lists of images for each label and their paths.

	Args:
	image_dir: String path to a folder containing subfolders of images.
	testing_percentage: Integer percentage of the images to reserve for tests.
	validation_percentage: Integer percentage of images reserved for validation.

	Returns:
	An OrderedDict containing an entry for each label subfolder, with images
	split into training, testing, and validation sets within each label.
	The order of items defines the class indices.
	"""
	MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1
	if not tf.gfile.Exists(image_dir):
		tf.logging.error("Image directory '" + image_dir + "' not found.")
		return None
	result = collections.OrderedDict()
	sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
	# The root directory comes first, so skip it.
	is_root_dir = True
	for sub_dir in sub_dirs:
		if is_root_dir:
			is_root_dir = False
			continue
		extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
								for ext in ['JPEG', 'JPG', 'jpeg', 'jpg']))
		file_list = []
		dir_name = os.path.basename(sub_dir)
		if dir_name == image_dir:
			continue
		tf.logging.info("Looking for images in '" + dir_name + "'")
		for extension in extensions:
	  		file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
	  		file_list.extend(tf.gfile.Glob(file_glob))
		if not file_list:
	  		tf.logging.warning('No files found')
	  		continue
		if len(file_list) < 20:
			tf.logging.warning(
			  'WARNING: Folder has less than 20 images, which may cause issues.')

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
			if percentage_hash < validation_percentage:
				validation_images.append(base_name)
			elif percentage_hash < (testing_percentage + validation_percentage):
				testing_images.append(base_name)
			else:
				training_images.append(base_name)
		result[label_name] = {
			'dir': dir_name,
			'training': training_images,
			'testing': testing_images,
			'validation': validation_images,
		}
	return result


def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


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

def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, module_name):
	"""Ensures all the training, testing, and validation bottlenecks are cached.

	Because we're likely to read the same image multiple times (if there are no
	distortions applied during training) it can speed things up a lot if we
	calculate the bottleneck layer values once for each image during
	preprocessing, and then just read those cached values repeatedly during
	training. Here we go through all the images we've found, calculate those
	values, and save them off.

	Args:
	sess: The current active TensorFlow Session.
	image_lists: OrderedDict of training images for each label.
	image_dir: Root folder string of the subfolders containing the training
	images.
	bottleneck_dir: Folder string holding cached files of bottleneck values.
	jpeg_data_tensor: Input tensor for jpeg data from file.
	decoded_image_tensor: The output of decoding and resizing the image.
	resized_input_tensor: The input node of the recognition graph.
	bottleneck_tensor: The penultimate output layer of the graph.
	module_name: The name of the image module being used.

	Returns:
	Nothing.
	"""
	how_many_bottlenecks = 0
	ensure_dir_exists(bottleneck_dir)
	for label_name, label_lists in image_lists.items():
		for category in ['training', 'testing', 'validation']:
	  		category_list = label_lists[category]
	  		for index, unused_base_name in enumerate(category_list):
		    	get_or_create_bottleneck(
			        sess, image_lists, label_name, index, image_dir, category,
			        bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
			        resized_input_tensor, bottleneck_tensor, module_name)

			    how_many_bottlenecks += 1
			    if how_many_bottlenecks % 100 == 0:
		      		tf.logging.info(
			          str(how_many_bottlenecks) + ' bottleneck files created.')
