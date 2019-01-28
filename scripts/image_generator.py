import os
import glob
import random


class ImageGenerator(object):
    def __init__(self,
                 cutoff = 50,
                 images_path = os.path.sep.join([os.environ['HOME'], 'flat_images'])):
        self.cutoff = cutoff
        self.all_fungi = self.generate_all_fungi_dict(images_path)
        self.long_tail = self.generate_long_tail()
        self.long = True


    def generate_all_fungi_dict(self, images_path):
        # generates a dictionary between a person and all the photos of that person
        all_fungi = {}
        for species_folder in os.listdir(images_path):
            species_photos = glob.glob(os.path.sep.join([images_path, species_folder , '*.JPG']))
            all_fungi[species_folder] = species_photos
        return all_fungi

    def generate_long_tail(self):
        lt = {k:len(v) for k,v in self.all_fungi.iteritems()}
        lt = {k: v>self.cutoff for k,v in lt.items()}
        tail = {}
        tail['long'] = [k for k,v in lt.items() if v == True]
        tail['short'] = [k for k,v in lt.items() if v == False]
        return tail

    def get_next_image_random(self):
        while True:
            # draw a species at random
            species = random.choice(self.all_fungi.keys())
            # draw a particular photo at random
            yield (random.choice(self.all_fungi[species]))

    def get_next_image_balanced(self):
        while True:
            if self.long:
                species = random.choice(self.long_tail['long'])
                self.long = False
            else:
                species = random.choice(self.long_tail['short'])
                self.long = True
            #picking photo at random from within that species
            yield (random.choice(self.all_fungi[species]))
