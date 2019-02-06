import random




def get_bottlenecks(image_dir, bottleneck_dir, image_lists,
					category, module_name, how_many):
	'''
	We need:

	image_dir - directory where all pictures are stored
	bottleneck_dir - directory where bottlenecks are stored
	image_lists - OrderedDict of train/validate/test images
	category - 'training','testing','validation'
	module_name - tf-hub name
	how_many - how many pictures we want to get (mini-batch size)
	'''

	class_count = len(image_lists.keys())
	bottlenecks = []
	ground_truths = []
	filenames = []
	# Retrieve a random sample of bottlenecks.
	for unused_i in range(how_many):
		#pick a species at random and save to label_lists
		label_index = random.randrange(class_count)
		label_name = list(image_lists.keys())[label_index]
		label_lists = image_lists[label_name]

		#obtain training/validation/testing section for that species
		category_list = label_lists[category]

		#get a random image for that species and train/val/test category
		image_index = random.randrange(len(category_list))
		base_name = category_list[image_index]
		sub_dir = label_lists['dir']
		image_name = os.path.join(image_dir, sub_dir, base_name)

		#makes sure directory is there so if we need to write to it it doesnt crash
		sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
		ensure_dir_exists(sub_dir_path)


		module_name = (module_name.replace('://', '~')  # URL scheme.
					 .replace('/', '~')  # URL and Unix paths.
					 .replace(':', '~').replace('\\', '~'))  # Windows paths.
		bottleneck_path = \
			os.path.join(bottleneck_dir, sub_dir, base_name) + '_' + module_name + '.txt'
		if not os.path.exists(bottleneck_path):
			tf.logging.info('Bottleneck file not found!')
		else:
			with open(bottleneck_path, 'r') as bottleneck_file:
				bottleneck_string = bottleneck_file.read()
			try:
				bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
				bottlenecks.append(bottleneck_values)
				ground_truths.append(label_index)
				filenames.append(image_name)
			except ValueError:
				tf.logging.warning('Bottleneck file was bad - skipping')
	return bottlenecks, ground_truths, filenames




def load_module(shared_layer_size = 1000, genus_count = 800, species_count = 1890, quantize_layer = 0, is_training = 0):
	'''
	We need:
	shared_layer_size - how big to make shared dense layer
	genus_count - how many classes for softmax classifier
	species_count - how many species are we classifying
	quantize_layer - for TFLite
	is_training - for evaluating only we don't need to create optimizer or loss
	'''
	FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
					  'FakeQuantWithMinMaxVarsPerChannel')
	module_name = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
	module_spec = hub.load_module_spec(module_name)
	m = hub.Module(module_spec)

	height, width = hub.get_expected_image_size(module_spec)
	resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])

	bottleneck_tensor = m(resized_input_tensor)
	with tf.Graph().as_default() as graph:
		wants_quantization = any(node.op in FAKE_QUANT_OPS
								 for node in graph.as_graph_def().node)

	#this will be fed later since during training its 0.5, and during eval 1
	keep_prob = tf.placeholder(tf.float32)
	batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
	assert batch_size is None, 'We want to work with arbitrary batch size.'
	with tf.name_scope('input'):
		bottleneck_input = tf.placeholder_with_default(
									bottleneck_tensor,
									shape=[batch_size, bottleneck_tensor_size],
									name='BottleneckInputPlaceholder')
		dropout_layer = tf.nn.dropout(bottleneck_input, keep_prob)



	# Organizing the following ops so they are easier to see in TensorBoard.
	with tf.name_scope('final_retrain_ops'):
		with tf.name_scope('weights'):
			initial_shared_value = tf.truncated_normal(
					[bottleneck_tensor_size, shared_layer_size], stddev=0.001)
			initial_genus_value = tf.truncated_normal(
					[shared_layer_size, genus_count], stddev=0.001)
			intial_species_value = tf.truncated_normal(
					[shared_layer_size, species_count], stddev=0.001)

			shared_layer_weights = tf.Variable(initial_shared_value,
			 										name='shared_layer_weights')
			genus_layer_weights = tf.Variable(initial_genus_value,
			 										name='final_genus_weights')
			species_layer_weights = tf.Variable(intial_species_value,
			 										name='final_species_weights')
			variable_summaries(shared_layer_weights)
			variable_summaries(genus_layer_weights)
			variable_summaries(species_layer_weights)

	with tf.name_scope('biases'):
		shared_layer_biases = tf.Variable(tf.zeros([shared_layer_size]),
		 											name='shared_layer_biases')
		layer_genus_biases = tf.Variable(tf.zeros([genus_count]),
		 											name='final_genus_biases')
		layer_species_biases = tf.Variable(tf.zeros([species_count]),
		 											name='final_species_biases')
		variable_summaries(shared_layer_biases)
		variable_summaries(layer_genus_biases)
		variable_summaries(layer_species_biases)

	with tf.name_scope('Wx_plus_b'):
		shared_logits = tf.matmul(dropout_layer, shared_layer_weights) + shared_layer_biases
		genus_logits = tf.matmul(shared_logits, genus_layer_weights) + layer_genus_biases
		species_logits = tf.matmul(shared_logits, species_layer_weights) + layer_species_biases
		#attach summaries for tensorboard
		tf.summary.histogram('pre_activations_shared', shared_logits)
		tf.summary.histogram('pre_activations_genus', genus_logits)
		tf.summary.histogram('pre_activations_species', species_logits)

	final_genus_tensor = tf.nn.softmax(genus_logits, name='final_genus_tensor')
	final_species_tensor = tf.nn.softmax(species_logits, name='final_species_tensor')

	tf.summary.histogram('activations_genus', final_genus_tensor)
	tf.summary.histogram('activations_species', final_species_tensor)

	# The tf.contrib.quantize functions rewrite the graph in place for
	# quantization. The imported model graph has already been rewritten, so upon
	# calling these rewrites, only the newly added final layer will be
	# transformed.
	if quantize_layer:
		if is_training:
		  tf.contrib.quantize.create_training_graph()
		else:
		  tf.contrib.quantize.create_eval_graph()



	# If this is an eval graph, we don't need to add loss ops or an optimizer.
	if not is_training:
		return (None, None, bottleneck_input,
		 		ground_truth_genus_input, ground_truth_species_input,
			  	final_genus_tensor, final_species_tensor)


	ground_truth_genus_input = tf.placeholder(
		tf.int64, [batch_size], name='GroundTruthGenusInput')
	ground_truth_species_input = tf.placeholder(
		tf.int64, [batch_size], name='GroundTruthSpeciesInput')
	with tf.name_scope('cross_entropy'):
		cross_entropy_mean_genus = tf.losses.sparse_softmax_cross_entropy(
									labels=ground_truth_genus_input,
									logits=genus_logits)
		cross_entropy_mean_species = tf.losses.sparse_softmax_cross_entropy(
									labels=ground_truth_species_input,
									logits=species_logits)
		combined_loss = cross_entropy_mean_genus + cross_entropy_mean_species

	tf.summary.scalar('cross_entropy', combined_loss)

	with tf.name_scope('train'):
		optimizer = tf.train.GradientDescentOptimizer(0.01)
		train_step = optimizer.minimize(combined_loss)

	return (train_step, combined_loss, bottleneck_input,
	 		ground_truth_genus_input, ground_truth_species_input,
		  	final_genus_tensor, final_species_tensor)
