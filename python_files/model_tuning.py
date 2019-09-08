"""helper functions to tune vgg16 and resnet50 models for transfer learning
"""

# import libaries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras_preprocessing.image import ImageDataGenerator

def create_train_test(label,pkl='../pickle_files/pic_df.pkl'):
    """convert pic_df.pkl from pickle_files folder into test and train pandas
       dataframe with specified classification label.
    # Arguments
        label: eyewear, hat, or beard, a string.
        pkl: filepath to pic_df.pkl from pickle_files folder.
    # Returns
        test and train pandas dataframe with the specified classification
        label.
    """
    # import pic_df
    df = pd.read_pickle(pkl)
    # subset the label dataframe
    sub_set = df[['pic_id',label]]
    X_train, X_test, y_train, y_test = train_test_split(sub_set['pic_id'],
                                                        sub_set[label],
                                                        stratify = sub_set[label],
                                                        test_size = 0.2
                                                        )
    df_train = pd.concat([X_train,y_train], axis=1)
    df_test = pd.concat([X_test,y_test], axis=1)
    return df_train, df_test


def create_generator(train_df, test_df,label,shuffle,batch_size,
                    rescale, preprocess_func, target_size,
                    class_mode, only_testing = False):
    """creates test and train generators for model.fit_generator.

    # Arguments
        train_df: train_df for model tuning.
        test_df: test_df for data validation tuning.
        label: eyewear, hat, or beard, a string.
        shuffle: shuffle the data sequence for each batch generator.
        batch_size: how many images per generator.
        rescale: 1./255 for vgg16 and None for resnet50.
        preprocess_func: vgg16_preprocess input or ResNet50 preprocess input.
        target_size: (150,150) for vgg16 and (224,224) for resnet50.
        class_mode: None or binary in this case.
        only_testing: model testing purpose or model tuning purpose, default
        is False.
    # Returns
        generators as a list. If class_mode is None, creates train and test
        generators but if class_model is binary, creates train, classweights,
        and test generators.
    """
    # initialize an empty list
    generators =[]
    # only generator traingenerator if it is for model tuning
    if not only_testing:
        traingen = ImageDataGenerator(
            rescale = rescale,
            zoom_range= [0.8,1.7],
            shear_range=0.2,
            brightness_range=[0.5,1.5],
            rotation_range = 40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            preprocessing_function=preprocess_func)

        train_generator = traingen.flow_from_dataframe(
            dataframe=train_df,
            directory='../data/pics',
            x_col='pic_id',
            y_col=label,
            batch_size=batch_size,
            shuffle = shuffle,
            target_size=target_size,
            class_mode = class_mode)
        generators.append(train_generator)
        if class_mode:
        # create classweights for train
            classweights = class_weight.compute_class_weight(
                'balanced',np.unique(train_generator.classes),
                train_generator.classes)
            generators.append(classweights)

    testgen = ImageDataGenerator(
        rescale = rescale,
        preprocessing_function=preprocess_func)

    test_generator = testgen.flow_from_dataframe(
        dataframe=test_df,
        directory='../data/pics',
        x_col='pic_id',
        y_col=label,
        batch_size=batch_size,
        shuffle=shuffle,
        target_size=target_size,
        class_mode=class_mode)
    generators.append(test_generator)

    return generators
