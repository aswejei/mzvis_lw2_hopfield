import ast
import math

import numpy as np
import random

IMAGE_RANGE = 5


def round(a):
    if a >= 0:
        return 1
    if a < 0:
        return -1


def activate(a):
    # res = 1 / (1 + pow(math.e, -a))
    # res = math.atan(a)
    res = math.tanh(a)
    # res = 1 if a >=0 else -1
    return res


class HopfieldNetwork:
    memorized_images: list
    __weight_matrix: list
    shuffled_neurons: list

    def __init__(self):
        self.__np_activation_fun = np.vectorize(activate)
        self.__np_round_fun = np.vectorize(round)
        self.memorized_images = []
        self.__weight_matrix = [[0 for _ in range(IMAGE_RANGE ** 2)] for _ in range(IMAGE_RANGE ** 2)]
        self.shuffled_neurons = [ind for ind in range(IMAGE_RANGE ** 2)]
        random.shuffle(self.shuffled_neurons)

    def memorize_model(self, model):
        print(print_image(model))
        self.memorized_images.append(model)
        s = np.matmul(np.array([model]).T, np.array([model]))
        self.__weight_matrix = (np.array(self.__weight_matrix) + s) \
            .tolist()
        for ind in range(len(self.__weight_matrix)):
            self.__weight_matrix[ind][ind] = 0

    def recognize_model(self, model, recognition_mode):
        print(print_image(model) + "\n")
        recognition_step = 0
        wm = np.array(self.__weight_matrix)
        # wm = np.array([[w / (IMAGE_RANGE ** 2) for w in ww] for ww in self.__weight_matrix])
        distorted_model = np.array(model).T
        relaxation_complete = False
        image_list = [np.array([]), np.array([])]
        output_im = np.array([0])
        while not relaxation_complete:
            recognition_step += 1
            if recognition_step > 5000:
                print("Image can\'t be recognized!\n")
                exit()
            input_im = distorted_model
            output_im_nonactivated = np.matmul(wm, distorted_model)
            if recognition_mode:
                if recognition_step == 1:
                    output_im = np.array(model).T
                random_neuron_index = recognition_step % (IMAGE_RANGE ** 2)
                neuron_index = self.shuffled_neurons[random_neuron_index]
                output_im[neuron_index] = output_im_nonactivated[neuron_index]
            else:
                output_im = output_im_nonactivated
            output_im = self.__np_activation_fun(output_im)
            distorted_model = output_im
            print(print_image(output_im) + "\n")
            # if np.array_equal(self.__np_round_fun(input_im), self.__np_round_fun(output_im)):
            if np.array_equal(input_im, output_im):
                relaxation_complete = True
            # if recognition_step > 2 and np.array_equal(self.__np_round_fun(distorted_model), self.__np_round_fun(image_list[recognition_step % 2])):
            if recognition_step > 2 and np.array_equal(distorted_model, image_list[recognition_step % 2]):
                relaxation_complete = True
            else:
                image_list[recognition_step % 2] = distorted_model

    @property
    def weight_matrix(self):
        return self.__weight_matrix

    @weight_matrix.setter
    def weight_matrix(self, value):
        self.__weight_matrix = value


def print_image(model):
    pic = ""
    for current_pixel in range(len(model)):
        if model[current_pixel] >= 0:
            pic += "#"
        elif model[current_pixel] < 0:
            pic += " "
        if (current_pixel + 1) % IMAGE_RANGE == 0:
            pic += "\n"
    return pic


def text_list_to_image_matrixes(raw_text):
    current_index_of_model_start = 0
    distinct_models = []
    for current_line in range(len(raw_text)):
        if raw_text[current_line] == '\n':
            new_model = raw_text[current_index_of_model_start:current_line]
            distinct_models.append(new_model)
            current_index_of_model_start = current_line + 1

    for model_index in range(len(distinct_models)):
        for line_index in range(len(distinct_models[0])):
            distinct_models[model_index][line_index] = distinct_models[model_index][line_index][:-1]
            vector_of_values = []
            for symbol in distinct_models[model_index][line_index]:
                if symbol == '1':
                    vector_of_values.append(1)
                if symbol == '0':
                    vector_of_values.append(-1)
            distinct_models[model_index][line_index] = vector_of_values

    flattened_distinct_models = []
    for model_index in range(len(distinct_models)):
        flattened_model = []
        for line_index in range(len(distinct_models[0])):
            flattened_model += distinct_models[model_index][line_index]
        flattened_distinct_models.append(flattened_model)
    return flattened_distinct_models


if __name__ == '__main__':
    work_mode = input(
        "Enter work mode of the program:\n1 - memorizing images \nWhatever else - image recognition\n\n")
    if work_mode == '1':
        filename = input("Enter filename of file with images to memorize: ")
        if filename[-4:] != ".txt":
            filename = filename + ".txt"
        with open(filename, "r") as file:
            line_list = file.readlines()
        distinct_models_list = text_list_to_image_matrixes(line_list)

        network = HopfieldNetwork()
        for md in distinct_models_list:
            network.memorize_model(md)

        wm_filename = input("Images are memorized!\nIf you don\'t want to save weight matrix, press Enter\n"
                            "If you do, enter name of the corresponding file\n\n")
        if wm_filename != "":
            if wm_filename[-4:] != ".txt":
                wm_filename = wm_filename + ".txt"
            wm_file = open(wm_filename, "w")
            wm_file.write(str(network.weight_matrix))
    else:
        weight_matrix_filename = input("Enter name of the file with weight matrix: ")
        if weight_matrix_filename[-4:] != ".txt":
            weight_matrix_filename = weight_matrix_filename + ".txt"
        with open(weight_matrix_filename, "r") as weight_matrix_file:
            weights = ast.literal_eval(weight_matrix_file.readline())

        distorted_image_filename = input("Enter name of the file with distorted image")
        if distorted_image_filename[-4:] != ".txt":
            distorted_image_filename = distorted_image_filename + ".txt"
        with open(distorted_image_filename, "r") as file:
            distorted_image_raw_text = file.readlines()
        distorted_image = text_list_to_image_matrixes(distorted_image_raw_text)

        network = HopfieldNetwork()
        network.weight_matrix = weights

        mode = input("Enter 1 if you want Hopfield network to work in synchronous mode\nEnter whatever else "
                     "if you want it to work asynchronously\n\n")
        if mode == "1":
            mode = False
        else:
            mode = True

        for image in distorted_image:
            network.recognize_model(image, mode)
            print("Recognition is finished!\n\n")
