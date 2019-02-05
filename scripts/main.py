import os
import sys
import argparse
import tensorflow as tf

from scripts.separate_imgs import separate_images_from_config
from scripts.model import MultiHead_Model
from scripts.image_generator import ImageSplitter, ImageGenerator, Dataset

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
    testing_percentage = config['data']['testing_percentage']
    validation_percentage = config['data']['validation_percentage']
    LONG_TAIL_CUTOFF = config['data']['LONG_TAIL_CUTOFF']
    #category
    balanced = config['training']['balanced']
    batch_size = config['training']['batch_size']
    prefetch_batch_buffer = config['training']['prefetch_batch_buffer']
    TRAINING_STEPS = config['training']['TRAINING_STEPS']
    EVALUATION_STEP = config['training']['EVALUATION_STEP']
    learning_rate = config['model']['learning_rate']

    '''
    Example values for debugging
    '''
    bottleneck_dir = '/Users/pablomartin/python/Mushroom_Classifier/bottlenecks'
    testing_percentage = 10
    validation_percentage = 10
    LONG_TAIL_CUTOFF = 150
    category = 'training'
    balanced = 0
    batch_size = 100
    prefetch_batch_buffer = 50
    learning_rate = 0.01
    TRAINING_STEPS = 10
    EVALUATION_STEP = 5

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
                                batch_size = batch_size,
                                prefetch_batch_buffer = prefetch_batch_buffer,
                                balanced = balanced)
    #validation tensorflow dataset
    validation_dataset = Dataset(generator = validation_gen,
                                batch_size = batch_size,
                                prefetch_batch_buffer = prefetch_batch_buffer,
                                balanced = balanced)

    training_dataset = Dataset(generator = testing_gen,
                                batch_size = batch_size,
                                prefetch_batch_buffer = prefetch_batch_buffer,
                                balanced = balanced)

    #let's load our model
    SPECIES_COUNT = Imagelists.species_classes
    GENUS_COUNT = Imagelists.genus_classes
    M = MultiHead_Model(shared_layer_size = 1000,
                        genus_count = GENUS_COUNT,
                        species_count = SPECIES_COUNT,
                        is_training = 1,
                        learning_rate =  0.01)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for unused_i in range(TRAINING_STEPS):
            bottlenecks, genus_truths, species_truths, _ = \
                                        sess.run(training_dataset.next_element)
            #training step
            sess.run(M.train_step, feed_dict = \
                                {M.bottleneck_input : bottlenecks,
                                 M.ground_truth_species_input : species_truths,
                                 M.ground_truth_genus_input: genus_truths,
                                 M.keep_prob : 0.5})

            if TRAINING_STEPS % EVALUATION_STEP == 0:
                '''
                We are evaluating accuracy on the training batch that we just
                trained on
                '''
                #evaluation step
                genus_accuracy, species_accuracy = \
                          sess.run([M.genus_evaluation_step,
                                    M.species_evaluation_step], feed_dict = \
                                    {M.bottleneck_input : bottlenecks,
                                     M.ground_truth_species_input : species_truths,
                                     M.ground_truth_genus_input: genus_truths,
                                     M.keep_prob : 1.0})
                '''
                Now we are drawing validation_batch_size pictures from the
                validation set and calculating accuracy, and NOT changing the
                weights of the network
                '''

                bottlenecks, genus_truths, species_truths, _ = \
                                        sess.run(validation_dataset.next_element)
                #evaluation step
                genus_accuracy, species_accuracy = \
                          sess.run([M.genus_evaluation_step,
                                    M.species_evaluation_step], feed_dict = \
                                    {M.bottleneck_input : bottlenecks,
                                     M.ground_truth_species_input : species_truths,
                                     M.ground_truth_genus_input: genus_truths,
                                     M.keep_prob : 1.0})

if __name__ == '__main__':
    main()
