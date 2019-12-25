import os
import sys
import pickle
import configparser
import tensorflow as tf
from datetime import datetime

from separate_imgs import separate_images_from_config
from model import MultiHead_Model
from image_generator import ImageSplitter, ImageGenerator, Dataset

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'


def main():

    config = configparser.ConfigParser()
    config.read('/home/ubuntu/Mushroom-Hikes/scripts/defaults.config')

    '''
    Values loaded from config file
    '''
    # directory paths
    BASE_DIR = config['files']['BASE_DIR']
    DATA_DIR = BASE_DIR + config['files']['DATA_DIR']
    DISCARDED_DATA_DIR = BASE_DIR + config['files']['DISCARDED_DATA_DIR']
    BOTTLENECK_DIR = BASE_DIR + config['files']['BOTTLENECK_DIR']
    SUMMARY_DIR = BASE_DIR + config['files']['SUMMARY_DIR']
    GRAPH_DIR = BASE_DIR + config['files']['GRAPH_DIR']

    # data separation parameters
    testing_percentage = int(config['data']['testing_percentage'])
    validation_percentage = int(config['data']['validation_percentage'])
    LONG_TAIL_CUTOFF = int(config['data']['LONG_TAIL_CUTOFF'])
    MIN_IMAGES_PER_CLASS = int(config['data']['MIN_IMAGES_PER_CLASS'])

    # training parameters
    training_batch_size = int(config['training']['training_batch_size'])
    validation_batch_size = int(config['training']['validation_batch_size'])
    prefetch_batch_buffer = int(config['training']['prefetch_batch_buffer'])
    balanced = int(config['training']['balanced'])
    TRAINING_STEPS = int(config['training']['TRAINING_STEPS'])
    EVALUATION_STEP = int(config['training']['EVALUATION_STEP'])
    INTERMEDIATE_FREQ = int(config['training']['INTERMEDIATE_FREQ'])

    # model parameters
    shared_layer_size = int(config['model']['shared_layer_size'])
    quantize_layer = int(config['model']['quantize_layer'])
    learning_rate = float(config['model']['learning_rate'])
    shared_layer_size = int(config['model']['shared_layer_size'])
    output_graph_target = config['model']['model_file']

    '''
    this will move classes that have less than MIN_IMAGES_PER_CLASS in config
    file to a side folder specified also in the defaults.config
    Only important if bottlenecks (image embeddings) have not been created
    '''
    separate_images_from_config(DATA_DIR, DISCARDED_DATA_DIR, MIN_IMAGES_PER_CLASS)

    Image_list_path = BASE_DIR + GRAPH_DIR + 'train_test_sets.p'
    if not os.path.exists(Image_list_path):
        # we must split dataset first, and feed that into different generators
        Imagelists = ImageSplitter(BOTTLENECK_DIR,
                                   testing_percentage,
                                   validation_percentage,
                                   LONG_TAIL_CUTOFF)
        pickle.dump(Imagelists, open(Image_list_path, 'wb'))
    else:
        Imagelists = pickle.load(open(Image_list_path, 'rb'))

    # gives us a random image file from the training set
    training_gen = ImageGenerator(imagesplitter=Imagelists,
                                  category='training',
                                  balanced=balanced)
    # gives us a random image file from the validation set
    validation_gen = ImageGenerator(imagesplitter=Imagelists,
                                    category='validation',
                                    balanced=balanced)
    # ditto for testing set
    testing_gen = ImageGenerator(imagesplitter=Imagelists,
                                 category='testing',
                                 balanced=balanced)

    # training tensorflow dataset
    training_dataset = Dataset(generator=training_gen,
                               batch_size=training_batch_size,
                               prefetch_batch_buffer=prefetch_batch_buffer)
    # validation tensorflow dataset
    validation_dataset = Dataset(generator=validation_gen,
                                 batch_size=validation_batch_size,
                                 prefetch_batch_buffer=prefetch_batch_buffer)

    testing_dataset = Dataset(generator=testing_gen,
                              batch_size=-1,
                              prefetch_batch_buffer=prefetch_batch_buffer)

    # let's load our model
    M = MultiHead_Model(shared_layer_size=shared_layer_size,
                        genus_count=Imagelists.genus_classes,
                        species_count=Imagelists.species_classes,
                        is_training=1,
                        learning_rate=learning_rate)

    train_saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(SUMMARY_DIR + 'train/', sess.graph)
        validation_writer = tf.summary.FileWriter(SUMMARY_DIR + 'validation/')

        '''
        We start training step. Each training step will go thru one batch size
        which we are setting to 1000, whilst our training set is ~700K
        So each epoch will be around ~700 TRAINING_STEPS
        '''
        for i in range(TRAINING_STEPS):
            bottlenecks, genus_truths, species_truths, _ = \
                sess.run(training_dataset.next_element)
            # training step
            train_summary, _ = sess.run([merged, M.train_step],
                                        feed_dict={
                M.bottleneck_input: bottlenecks,
                M.ground_truth_species_input: species_truths,
                M.ground_truth_genus_input: genus_truths,
                M.keep_prob: 0.5})
            train_writer.add_summary(train_summary, i)

            is_last_step = (i + 1 == TRAINING_STEPS)
            if (i % EVALUATION_STEP) == 0 or is_last_step:
                '''
                We are evaluating accuracy on the training batch that we just
                trained on with dropout of 1!
                '''
                # evaluation step
                train_genus_accuracy, train_species_accuracy = \
                    sess.run([M.genus_evaluation_step,
                              M.species_evaluation_step], feed_dict={M.bottleneck_input: bottlenecks,
                                                                     M.ground_truth_species_input: species_truths,
                                                                     M.ground_truth_genus_input: genus_truths,
                                                                     M.keep_prob: 1.0})
                '''
                Now we are drawing validation_batch_size pictures from the
                validation set and calculating accuracy, and not updating the
                weights of the network
                '''

                bottlenecks, genus_truths, species_truths, _ = \
                    sess.run(validation_dataset.next_element)
                # evaluation step
                validation_summary, validate_genus_accuracy, validate_species_accuracy = \
                    sess.run([merged, M.genus_evaluation_step,
                              M.species_evaluation_step], feed_dict={M.bottleneck_input: bottlenecks,
                                                                     M.ground_truth_species_input: species_truths,
                                                                     M.ground_truth_genus_input: genus_truths,
                                                                     M.keep_prob: 1.0})
                validation_writer.add_summary(validation_summary, i)
                tf.logging.info('%s: Step %d: Train \nGenus accuracy = %.1f%% \nSpecies accuracy = %.1f%%'
                                % (datetime.now(), i, train_genus_accuracy * 100, train_species_accuracy * 100))
                tf.logging.info('%s: Step %d: Validate \nGenus accuracy = %.1f%% \nSpecies accuracy = %.1f%%'
                                % (datetime.now(), i, validate_genus_accuracy * 100, validate_species_accuracy * 100))

            if (INTERMEDIATE_FREQ > 0 and (i % INTERMEDIATE_FREQ == 0) and i > 0):
                # If we want to do an intermediate save, save a checkpoint of the train
                # graph, to restore into the eval graph.
                #train_saver.save(sess, CHECKPOINT_NAME)
                intermediate_file_name = (GRAPH_DIR + 'intermediate_'
                                          + str(i) + '.pb')
                train_saver.save(sess, CHECKPOINT_NAME)
                tf.logging.info('Save intermediate result to : ' +
                                intermediate_file_name)
                output_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess, sess.graph.as_graph_def(),
                    ['final_genus_tensor', 'final_species_tensor'])
                with tf.gfile.FastGFile(intermediate_file_name, 'wb') as f:
                    f.write(output_graph_def.SerializeToString())

        '''
        Now that training is done we will run our final evaluation step on the
        entire test set
        '''
        train_saver.save(sess, CHECKPOINT_NAME)  # save before we do this
        bottlenecks, genus_truths, species_truths, _ = \
            sess.run(testing_dataset.next_element)

        # evaluation step
        test_genus_accuracy, test_species_accuracy = \
            sess.run([M.genus_evaluation_step,
                      M.species_evaluation_step], feed_dict={M.bottleneck_input: bottlenecks,
                                                             M.ground_truth_species_input: species_truths,
                                                             M.ground_truth_genus_input: genus_truths,
                                                             M.keep_prob: 1.0})
        genus_softmax, species_softmax = \
            sess.run([M.final_genus_tensor,
                      M.final_species_tensor],
                     feed_dict={M.bottleneck_input: bottlenecks,
                                M.keep_prob: 1.0})

        tf.logging.info('----------TEST SET EVAL ----------')
        tf.logging.info('Genus accuracy = %.1f%% Species accuracy = %.1f%%'
                        % (test_genus_accuracy * 100, test_species_accuracy * 100))
        tf.logging.info('----------TEST SET EVAL ----------')

        '''
        Now we are calculating top-5 and top-10 on the testing set
        '''

        gtop_accuracy = [[], [], []]
        stop_accuracy = [[], [], []]
        for gvec, svec, gtruth, struth in \
                zip(genus_softmax, species_softmax, genus_truths, species_truths):
            gtop_10 = [np.where(gvec == w)[0][0] for w in sorted(gvec)[-10:]]
            stop_10 = [np.where(svec == w)[0][0] for w in sorted(svec)[-10:]]
            gtop_accuracy[0].append(int(gtruth in gtop_10[-1:]))
            gtop_accuracy[1].append(int(gtruth in gtop_10[-5:]))
            gtop_accuracy[2].append(int(gtruth in gtop_10[-10:]))
            stop_accuracy[0].append(int(struth in stop_10[-1:]))
            stop_accuracy[1].append(int(struth in stop_10[-5:]))
            stop_accuracy[2].append(int(struth in stop_10[-10:]))

        Gtotal_accuracy = np.mean(gtop_accuracy[0])
        Stotal_accuracy = np.mean(stop_5_accuracy)
        for gacc, sacc, KK in zip(gtop_accuracy, stop_accuracy, [1, 5, 10]):
            tf.logging.info('Genus top %i accuracy: %1.2f%%'
                            % (KK, np.mean(gacc) * 100))
            tf.logging.info('Species top %i accuracy: %1.2f%%'
                            % (KK, np.mean(sacc) * 100))

        '''
        Save final graph to disk!
        '''
        output_graph_target = BASE_DIR + GRAPH_DIR + output_graph_target
        tf.logging.info('Save final graph to : ' + output_graph_target)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(),
            ['final_genus_tensor', 'final_species_tensor'])
        with tf.gfile.FastGFile(output_graph_target, 'wb') as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    main()
