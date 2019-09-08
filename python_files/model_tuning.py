"""helper functions to tune vgg16 and resnet50 models for transfer learning
"""

# import libaries
import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint

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
        generators as a tuple. If class_mode is None, creates train and test
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

    return tuple(generators)


def save_bottleneck_features(label,model_type, preprocess_func,rescale,
                             target_size,file_path1,file_path2):
    """save output of model features from the vgg16 or resnet50 non dense
       layers as npy files in file_path1 and file_path2. Returns corresponding
       test and train df for model tuning.
    # Arguments
        label: eyewear, hat, or beard, a string.
        model_type:ResNet50 or VGG16.
        preprocess_func: vgg16_preprocess input or ResNet50 preprocess input.
        rescale: 1./255 for vgg16 and None for resnet50.
        target_size: (150,150) for vgg16 and (224,224) for resnet50.
        file_path1: folder path to save train data npy.
        file_path2: folder path to save test data npy.
    # Returns
        train and test df for fully connected layer tuning.
    """
    # intialize the model, vgg16 or ResNet50.
    # make sure not to train the top layers.
    base_model = model_type(weights = 'imagenet',include_top = False)
    # generate test_train df.
    train_df, test_df = create_train_test(label)
    # create train_generator and test_generator to get bottleneck inputs for
    # train and test df.
    # make sure shuffle is False so we know the label follows the sequence of
    # the dataframe so we can tune top_model and class_mode is None.
    generators = create_generator(train_df=train_df,
                                  test_df=test_df,
                                  label=label,
                                  shuffle=False,
                                  rescale=rescale,
                                  preprocess_func=preprocess_func,
                                  batch_size=16,
                                  target_size=target_size,
                                  class_mode=None)
    # in this case we know the class_mode is None
    train_generator, test_generator = generators

    # get features saved as .npy in file_path1 and file_path2
    bottleneck_features_train = base_model.predict_generator(
        train_generator, train_df.shape[0]//16)
    np.save(open(file_path1,'wb'),
           bottleneck_features_train)

    bottleneck_features_test = base_model.predict_generator(
        test_generator, test_df.shape[0]//16)
    np.save(open(file_path2,'wb'),
           bottleneck_features_test)
    return train_df, test_df


def resnet50_model(input_shape, dropout=0.25):
    """ create dense layer for resnet50 model.
    # Arguments
        input_shape: input_shape for pooling layer.
        dropout:percentage for Dropout layer to prevent overfitting, default
        is 0.25.
    # Returns
        fully connected top model for resnet50.
    """
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=input_shape))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='sigmoid'))
    return model


def vgg16_model(input_shape,dropout=0.5):
    """ create dense layer for vgg16 model.
    # Arguments
        input_shape: input_shape for flatten layer.
        dropout:percentage for Dropout layer to prevent overfitting, default
        is 0.5.
    # Returns
        fully connected top model for vgg16.
    """
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train_top_model(train_df, test_df, epoch, label, model_type, rescale,
                    preprocess_func,target_size, file_path1,file_path2,
                    file_path3):
    """ train fully connected top model and save best weights to help fine
        tuning models when we unfreeze additional layers.
    # Arguments
        train_df: dataframe returned from save_bottleneck_features function.
        test_df: dataframe returned from save_bottleneck_features function.
        epoch: number of epochs in model tuning.
        label: eyewear, hat, or beard, a string.
        model_type: resnet50 or vgg16, a string.
        rescale: 1./255 for vgg16 and None for resnet50.
        preprocess_func: vgg16_preprocess input or ResNet50 preprocess input.
        target_size: (150,150) for vgg16 and (224,224) for resnet50.
        file_path1: folder path to train data npy.
        file_path2: folder path to test data npy.
        file_path3: folder path to save top model best weights h5.
    """
    # retrieve train and test
    train_data = np.load(open(file_path1,'rb'))
    # make sure train_data and train_label have same num of samples
    # convert string label to 1 or 0 for model tuning
    train_label = np.array(
        train_df[label].map({'0':0, '1':1})
    )[:-(train_df.shape[0]%16)]

    test_data = np.load(open(file_path2,'rb'))
    test_label = np.array(
        test_df[label].map({'0':0, '1':1})
    )[:-(test_df.shape[0]%16)]

    # build top model
    if model_type == 'resnet50':
        model = resnet50_model(train_data.shape[1:])
    if model_type == 'vgg16':
        model = vgg16_model(train_data.shape[1:])

    model.compile(optimizer=optimizers.Adam(),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    # checkpoint for best weights
    checkpoint = ModelCheckpoint(file_path3,
        monitor='val_acc', verbose=1, save_best_only=True, mode='max'
        )
    callbacks_list = [checkpoint]

    _, classweight, _  = create_generator(train_df, test_df,label,False,16,
                    rescale, preprocess_func, target_size,
                    'binary', only_testing = False)

    model.fit(train_data, train_label,
             epochs=epoch,
             batch_size=16,
             validation_data=(test_data,test_label),
             callbacks=callbacks_list,
             class_weight = classweight)
    # clears the model to enable next model tuning
    del model
    keras.backend.clear_session()


def fine_tune_model(train_df, test_df,epoch, label, model_type,dim, rescale,
                    preprocess_func, file_path3, file_path4):
    """ Fine tunes the model in addition to the top model. Both vgg16 and
        resnet50 goes 4 layers up in addition to the dense layer.
    # Arguments
        train_df: dataframe returned from save_bottleneck_features function.
        test_df: dataframe returned from save_bottleneck_features function.
        epoch: number of epochs in model tuning.
        label: eyewear, hat, or beard, a string.
        model_type: resnet50 or vgg16, a string.
        dim: 150 for vgg16 and 224 for resnet50
        rescale: 1./255 for vgg16 and None for resnet50.
        preprocess_func: vgg16_preprocess input or ResNet50 preprocess input.
        file_path3: folder path to top model best weights h5.
        file_path4: folder path to save fine_tune model best weights h5.
    # Returns
        model results
    """
    # build model and freeze top layers
    if model_type == 'resnet50':
        base_model = ResNet50(weights='imagenet',include_top=False,
                              input_shape=(dim,dim,3))
        # build top model
        top_model = resnet50_model(base_model.output_shape[1:],0.25)
    if model_type == 'vgg16':
        base_model = VGG16(weights='imagenet',include_top=False,
                           input_shape=(dim,dim,3))
        top_model = vgg16_model(base_model.output_shape[1:],0.5)
    # load saved weights to fine tune parameters
    top_model.load_weights(file_path3)
    # add top model to model
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    # we will tune last 5 layers of the model for both vgg16 and resnet50
    for layer in model.layers[:-5]:
        layer.trainable = False
    # we can tune the parameters for lr and momentum later to get better
    # results
    model.compile(loss='binary_crossentropy',
             optimizer=optimizers.SGD(lr=5e-5, momentum = 0.9),
             metrics=['accuracy'])
    # prepare train generator using data augmentation to battle small
    # sample size
    generators = create_generator(train_df,test_df, label,True,16,rescale,
                                                    preprocess_func,(dim,dim),
                                                    'binary')
    train_generator, classweight, test_generator = generators
    # checkpoint for best weights
    checkpoint = ModelCheckpoint(file_path4, monitor='val_acc', verbose=1,
                                save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # run and fit model
    result = model.fit_generator(
    train_generator,
    steps_per_epoch=train_df.shape[0]//16,
    epochs=epoch,
    validation_data=test_generator,
    validation_steps=test_df.shape[0]//16,
    verbose=1,class_weight=list(classweight),
    callbacks=callbacks_list
    )
   # clears the model to enable next model tuning
    del model
    keras.backend.clear_session()
    return result
