#!/usr/bin/env python3
"""Executable file for recommmending products to IG profiles.
"""

# import libaries
import os
# disable tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
from keras_preprocessing.image import load_img, img_to_array
from keras.models import model_from_json
from keras.applications.resnet50 import preprocess_input as rs50
from vgg16_preprocess import preprocess_input as vgp
from PIL import Image as pil_image
#~ import warnings
#~ warnings.filterwarnings('ignore')
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session

# disable tensorflow logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# allows the computer to use GPU to run the model
config = tf.ConfigProto()
# dynamically grow the memory used on the GPU, necessary for RTX cards
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = False
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)
keras.backend.get_session().run(tf.global_variables_initializer())

def cropping(file_path):
    """Crops faces from a pic.
    # Arguments
        file_path: file_path to each pic.
    # Returns
        a list of cropped images of faces for each pic.
    """
    # load pic
    pic = face_recognition.load_image_file(file_path)
    # obtain list of images with tuples, 4 points
    # (ymin,xmax,ymax,xmin)
    face_locations = face_recognition.face_locations(pic)
    # intialize an empty list of available faces
    faces = []
    for loc in face_locations:
        delta_y = loc[2] - loc[0]
        delta_x = loc[1] - loc[3]
        # experimented to find a good width for hats and beard
        y_width = 2.5*delta_y
        x_increase = int((y_width*0.7 - delta_x)/2)
        y_increase = 1.25*delta_y/2
        y_min = loc[0]-int(y_increase)
        y_max = loc[2]+int(y_increase*0.4)
        x_min = loc[3]-x_increase
        x_max = loc[1]+x_increase
        # make sure not to go out of range
        if x_min < 0:
            x_min = 0
        if x_max > pic.shape[1]:
            x_max = pic.shape[1]
        if y_min < 0:
            y_min = 0
        if y_max > pic.shape[0]:
            y_max = pic.shape[0]
        faces.append(pic[y_min:y_max,x_min:x_max, :])
    return faces


def load_pretrained_models(model_path,weight_path):
    """Load pretrained models for final recommendation.
    # Arguments
         model_path: json model path for resnet50 or vgg16.
         weight_path: file path to saved best weights as h5.
    # Returns
        saved models.
    """
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load model weight
    loaded_model.load_weights(weight_path)
    return loaded_model


def recommend(filepath):
    """recommends most relevant product to each IG user picture folder.
       Products are eyewear, hat, and beard.
    # Arguments
         filepath: IG user picture folder file path.
    """
    files = os.listdir(filepath)
    # keep track of num eyewear, beard, hat, and total pic with faces cropped
    eyewear = 0
    beard = 0
    hat = 0
    total = 0
    # sequence of eyewear, beard, and hat
    highest_score =[0,0,0]
    highest_image =['','','']
    for file in files:
        if file != '.DS_Store':
        # get face locations
            faces = cropping(os.path.join(filepath,file))
            if not faces:
                continue
            total += 1
            for face in faces:
                # remove identified false faces
                # assume the fake faces are less than 150 pixel
                if face.shape[0] <=150 or face.shape[1] <= 150:
                    continue
                print(f'Analyzing image number {total}...')
                # get predicted value for both resnet50 and vgg16
                # vgg16 predictions first:
                img_v = pil_image.fromarray(face).resize((150,150),
                                            pil_image.NEAREST)
                img_v = np.expand_dims(img_v,axis=0)
                img_v = vgp(img_v)
                img_v=img_v/255
                # follow this exact sequence: eyewear, beard, hat
                preds_v = []
                for model in [vgg_eyewear,vgg_beard,vgg_hat]:
                    pred_v= model.predict(img_v)[0][0]
                    preds_v.append(pred_v)
                # resnet50 predictions
                img_r = pil_image.fromarray(face).resize((224,224),
                                            pil_image.NEAREST)
                img_r = np.expand_dims(img_r,axis=0)
                img_r = rs50(img_r)
                preds_r = []
                for model in [resnet_eyewear,resnet_beard,resnet_hat]:
                    pred_r = model.predict(img_r)[0][0]
                    preds_r.append(pred_r)
                # average the two value as an ensemble method
                pred_c = np.array([preds_v, preds_r]).mean(axis=0)
                # update highest scores
                # decide if each image has eyewear, beard or hat
                if pred_c[0] >= 0.5:
                    eyewear += 1
                if pred_c[0] > highest_score[0]:
                    highest_image[0] = os.path.join(filepath,file)
                    highest_score[0] = pred_c[0]
                if pred_c[1] >= 0.5:
                    beard += 1
                if pred_c[1] > highest_score[1]:
                    highest_image[1] = os.path.join(filepath,file)
                    highest_score[1] = pred_c[1]
                if pred_c[2] >= 0.5:
                    hat += 1
                if pred_c[2] > highest_score[2]:
                    highest_image[2] = os.path.join(filepath,file)
                    highest_score[2] = pred_c[2]
    # recommend products based upon most percentage per total num of pics
    index = np.argmax([eyewear/total, beard/total, hat/total])
    product = ['eyewear','beard','hat'][index]
    img_chosen = highest_image[index]
    img = load_img(img_chosen, target_size = (512,512))
    img = img_to_array(img)
    img = img/255
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"We recommend {product} products to this IG user!")
    plt.show()


# initialize the 6 models outside the recommend function to speed up process:
model_path_v = '../tuning_data/VGG_16_tuning/vgg_model.json'
file_path_v = '../tuning_data/VGG_16_tuning/'
eyewear_v = file_path_v+'best_vgg16_model_eyewear.h5'
hat_v = file_path_v+'best_vgg16_model_hat.h5'
beard_v = file_path_v+'best_vgg16_model_beard.h5'
vgg_eyewear = load_pretrained_models(model_path_v,eyewear_v)
vgg_hat = load_pretrained_models(model_path_v,hat_v)
vgg_beard = load_pretrained_models(model_path_v,beard_v)
model_path_r = '../tuning_data/resnet_data/resnet50_model_5_up.json'
file_path_r = '../tuning_data/resnet_data/untracked_resnet50/'
eyewear_r = file_path_r+'best_resnet50_model_eyewear.h5'
hat_r = file_path_r+'best_resnet50_model_hat.h5'
beard_r = file_path_r+'best_resnet50_model_beard.h5'
resnet_eyewear = load_pretrained_models(model_path_r,eyewear_r)
resnet_hat = load_pretrained_models(model_path_r,hat_r)
resnet_beard = load_pretrained_models(model_path_r,beard_r)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Wrong input. Please enter a folder path.')
        sys.exit(1)
    filepath = sys.argv[1]
    recommend(filepath)
