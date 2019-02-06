import os
import sys
import argparse
import tensorflow as tf
from datetime import datetime

from scripts.separate_imgs import separate_images_from_config
from scripts.model import MultiHead_Model
from scripts.image_generator import ImageSplitter, ImageGenerator, Dataset

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'

def main():

    config = configparser.ConfigParser()
    config.read('/home/ubuntu/Mushroom-Hikes/scripts/defaults.config')
    '''
    this will separate images before creating bottlenecks - not as important
    now that they have been precomputed
    '''
    separate_images_from_config(config)

    '''
    Values loaded from config file
    '''
    bottleneck_dir = config['files']['BOTTLENECK_DIR']
    summary_dir = config['files']['SUMMARY_DIR']
    testing_percentage = config['data']['testing_percentage']
    validation_percentage = config['data']['validation_percentage']
    LONG_TAIL_CUTOFF = config['data']['LONG_TAIL_CUTOFF']
    #category
    balanced = config['training']['balanced']
    batch_size = config['training']['batch_size']
    prefetch_batch_buffer = config['training']['prefetch_batch_buffer']
    TRAINING_STEPS = config['training']['TRAINING_STEPS']
    EVALUATION_STEP = config['training']['EVALUATION_STEP']
    INTERMEDIATE_FREQ = config['training']['INTERMEDIATE_FREQ']

    learning_rate = config['model']['learning_rate']
    shared_layer_size = config['model']['shared_layer_size']
    '''
    Example values for debugging
    '''
    bottleneck_dir = '/Users/pablomartin/python/Mushroom_Classifier/bottlenecks'
    summary_dir = 'tensorboard_summaries/'
    output_graph_target = 'output_graph.pb'
    testing_percentage = 10
    validation_percentage = 10
    LONG_TAIL_CUTOFF = 150
    category = 'training'
    balanced = 0
    training_batch_size = 1000
    validation_batch_size = 1000
    prefetch_batch_buffer = 100
    learning_rate = 0.01
    shared_layer_size = 1000

    TRAINING_STEPS = 10
    EVALUATION_STEP = 5
    INTERMEDIATE_FREQ = 700

    #we must split dataset first, and feed that into different generators
    Imagelists = ImageSplitter(bottleneck_dir,
                               testing_percentage,
                               validation_percentage,
                               LONG_TAIL_CUTOFF)


    #gives us a random image file from the training set
    training_gen = ImageGenerator(imagesplitter = Imagelists,
                                  category = 'training',
                                  balanced = balanced)
    #gives us a random image file from the validation set
    validation_gen = ImageGenerator(imagesplitter = Imagelists,
                                    category = 'validation',
                                    balanced = balanced)
    #ditto for testing set
    testing_gen = ImageGenerator(imagesplitter = Imagelists,
                                    category = 'testing',
                                    balanced = balanced)

    #training tensorflow dataset
    training_dataset = Dataset(generator = training_gen,
                                batch_size = training_batch_size,
                                prefetch_batch_buffer = prefetch_batch_buffer,
                                balanced = balanced)
    #validation tensorflow dataset
    validation_dataset = Dataset(generator = validation_gen,
                                batch_size = validation_batch_size,
                                prefetch_batch_buffer = prefetch_batch_buffer,
                                balanced = balanced)

    testing_dataset = Dataset(generator = testing_gen,
                                batch_size = -1,
                                prefetch_batch_buffer = prefetch_batch_buffer,
                                balanced = balanced)

    #let's load our model
    M = MultiHead_Model(shared_layer_size = shared_layer_size,
                        genus_count = Imagelists.genus_classes,
                        species_count = Imagelists.species_classes,
                        is_training = 1,
                        learning_rate =  0.01)

    train_saver = tf.train.Saver()
    init = tf.global_variables_initializer()


    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     bottlenecks, genus_truths, species_truths, _ = \
    #                                 sess.run(training_dataset.next_element)
    #     #training step
    #     GCP, SCP = sess.run([M.final_genus_tensor, M.final_species_tensor], feed_dict = \
    #                         {M.bottleneck_input : bottlenecks,
    #                          M.ground_truth_species_input : species_truths,
    #                          M.ground_truth_genus_input: genus_truths,
    #                          M.keep_prob : 1.0})


    with tf.Session() as sess:
        sess.run(init)
        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summary_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(summary_dir + '/validation')


        '''
        We start training step. Each training step will go thru one batch size
        which we are setting to 1000, whilst our training set is ~700K
        So each epoch will be around ~700 TRAINING_STEPS
        '''
        for i in range(TRAINING_STEPS):
            bottlenecks, genus_truths, species_truths, _ = \
                                        sess.run(training_dataset.next_element)
            #training step
            train_summary, _ = sess.run([merged, M.train_step],
                                feed_dict = { \
                                 M.bottleneck_input : bottlenecks,
                                 M.ground_truth_species_input : species_truths,
                                 M.ground_truth_genus_input: genus_truths,
                                 M.keep_prob : 0.5})
            train_writer.add_summary(train_summary, i)

            is_last_step = (i + 1 == TRAINING_STEPS)
            if (i % EVALUATION_STEP) == 0 or is_last_step:
                '''
                We are evaluating accuracy on the training batch that we just
                trained on
                '''
                #evaluation step
                train_genus_accuracy, train_species_accuracy = \
                          sess.run([M.genus_evaluation_step,
                                    M.species_evaluation_step], feed_dict = \
                                    {M.bottleneck_input : bottlenecks,
                                     M.ground_truth_species_input : species_truths,
                                     M.ground_truth_genus_input: genus_truths,
                                     M.keep_prob : 1.0})
                '''
                Now we are drawing validation_batch_size pictures from the
                validation set and calculating accuracy, and not updating the
                weights of the network
                '''

                bottlenecks, genus_truths, species_truths, _ = \
                                        sess.run(validation_dataset.next_element)
                #evaluation step
                validate_genus_accuracy, validate_species_accuracy = \
                          sess.run([M.genus_evaluation_step,
                                    M.species_evaluation_step], feed_dict = \
                                    {M.bottleneck_input : bottlenecks,
                                     M.ground_truth_species_input : species_truths,
                                     M.ground_truth_genus_input: genus_truths,
                                     M.keep_prob : 1.0})
                tf.logging.info('%s: Step %d: Train \nGenus accuracy = %.1f%% \nSpecies accuracy = %.1f%%'
                    %(datetime.now(), i, train_genus_accuracy * 100, train_species_accuracy * 100))
                tf.logging.info('%s: Step %d: Validate \nGenus accuracy = %.1f%% \nSpecies accuracy = %.1f%%'
                    %(datetime.now(), i, validate_genus_accuracy * 100, validate_species_accuracy * 100))

            if (INTERMEDIATE_FREQ > 0 and (i % INTERMEDIATE_FREQ == 0) and i > 0):
                # If we want to do an intermediate save, save a checkpoint of the train
                # graph, to restore into the eval graph.
                train_saver.save(sess, CHECKPOINT_NAME)
                intermediate_file_name = ('intermediate_graphs/intermediate_' \
                                                               + str(i) + '.pb')
                tf.logging.info('Save intermediate result to : ' +
                                intermediate_file_name)
                output_graph_def = tf.graph_util.convert_variables_to_constants(
                                sess, sess.graph.as_graph_def(),
                                ['final_genus_tensor','final_species_tensor'])
                with tf.gfile.FastGFile(intermediate_file_name, 'wb') as f:
                    f.write(output_graph_def.SerializeToString())

        '''
        Now that training is done we will run our final evaluation step on the
        entire test set
        '''
        bottlenecks, genus_truths, species_truths, _ = \
                                sess.run(testing_dataset.next_element)

        #evaluation step
        test_genus_accuracy, test_species_accuracy = \
                  sess.run([M.genus_evaluation_step,
                            M.species_evaluation_step], feed_dict = \
                            {M.bottleneck_input : bottlenecks,
                             M.ground_truth_species_input : species_truths,
                             M.ground_truth_genus_input: genus_truths,
                             M.keep_prob : 1.0})

        tf.logging.info('----------TEST SET EVAL ----------')
        tf.logging.info('Genus accuracy = %.1f%% \nSpecies accuracy = %.1f%%'
                    %(test_genus_accuracy * 100, test_species_accuracy * 100))
        tf.logging.info('----------TEST SET EVAL ----------')

        tf.logging.info('Save final graph to : ' + output_graph_target)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                        sess, sess.graph.as_graph_def(),
                        ['final_genus_tensor','final_species_tensor'])
        with tf.gfile.FastGFile(output_graph_target, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    main()
