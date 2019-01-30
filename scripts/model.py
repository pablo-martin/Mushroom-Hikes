import tensorflow as tf
import tensorflow_hub as hub

module_spec = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"
module = hub.load_module_spec(module_spec)
height, width = hub.get_expected_image_size(module)
n_words = 1000
dim_embed = 100
dim_hidden = 1000
n_lstm_steps = 5



with tf.Graph().as_default() as graph:
	resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
	m = hub.Module(module_spec)
	bottleneck_tensor = m(resized_input_tensor)
	batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()

	#word embedding inputs
	word_embedding = tf.Variable(tf.random_uniform([n_words, dim_embed], -0.1, 0.1), name='word_embedding')
	embedding_bias = tf.Variable(tf.zeros([dim_embed]), name='embedding_bias')

	#define LSTM
	lstm = tf.contrib.rnn.LSTMCell(dim_hidden)

	#image embedding inputs
	img_embedding = tf.Variable(tf.random_uniform([bottleneck_tensor_size, dim_hidden], -0.1, 0.1), name='img_embedding')
	img_embedding_bias = tf.Variable(tf.zeros([dim_hidden]), name='img_embedding_bias')

	#LSTM output to word embedding output
	word_encoding = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='word_encoding')
	word_encoding_bias = tf.Variable(tf.zeros([n_words]), name='word_encoding_bias')

	# (describes how long our caption is with an array of 0/1 values of length `maxlen`
	img = tf.placeholder(tf.float32, [batch_size, bottleneck_tensor_size])
	caption_placeholder = tf.placeholder(tf.int32, [batch_size, n_lstm_steps])
	mask = tf.placeholder(tf.float32, [batch_size, n_lstm_steps])

	# getting an initial LSTM embedding from our image_imbedding
	image_embedding = tf.matmul(img, img_embedding) + img_embedding_bias

	# setting initial state of our LSTM
	state = lstm.zero_state(batch_size, dtype=tf.float32)

	total_loss = 0.0
	with tf.variable_scope("RNN"):
		for i in range(n_lstm_steps):
			if i > 0:
				'''if this isnt the first iteration of our LSTM we need to get
				the word_embedding corresponding to the (i-1)th word in our caption'''
				with tf.device("/cpu:0"):
					current_embedding = \
						tf.nn.embedding_lookup(word_embedding,
						 						caption_placeholder[:,i-1]) \
															+ embedding_bias
			else:
				 #if this is the first iteration of our LSTM we utilize the embedded image as our input
				current_embedding = image_embedding
			if i > 0:
				# allows us to reuse the LSTM tensor variable on each iteration
				tf.get_variable_scope().reuse_variables()

			out, state = lstm(current_embedding, state)


			if i > 0:
				#get the one-hot representation of the next word in our caption
				labels = tf.expand_dims(caption_placeholder[:, i], 1)
				ix_range=tf.range(0, batch_size, 1)
				ixs = tf.expand_dims(ix_range, 1)
				concat = tf.concat([ixs, labels],1)
				onehot = tf.sparse_to_dense(
						concat, tf.stack([batch_size, n_words]), 1.0, 0.0)


				#perform a softmax classification to generate the next word in the caption
				logit = tf.matmul(out, word_encoding) + word_encoding_bias
				cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)
				cross_entropy = cross_entropy * mask[:,i]

				loss = tf.reduce_sum(cross_entropy)
				total_loss += loss

		total_loss = total_loss / tf.reduce_sum(mask[:,1:])
