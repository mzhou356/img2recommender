"""model class BaseClassifier that tunes both VGG16 and ResNet50 and prints
   evaluation reports.
"""
# import libaries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)
import keras
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras_preprocessing.image import ImageDataGenerator

class BaseClassifier:
    """Performs VGG16 and ResNet Model Tuning and prints out evaluations.
    """

    def __init__(self, label, rescale, preprocess_func, dim, modelname,
                 modeltype, pkl='../pickle_files/pic_df.pkl'):
        """initialization function for BaseClassifier.

        Parameters
        ----------
        param1 : label
            eyewear, hat, or beard, a string.
        param2 : rescale
            1./255 for vgg16 and None for resnet50.
        param3 : preprocess_func
            vgg16_preprocess input or ResNet50 preprocess input.
        param4: dim
            150 for vgg16 and 224 for resnet50.
        param5: modelname
            ResNet50 or VGG16.
        param6: modeltype
            resnet50 or vgg16, a string.
        param7: pkl
            file path to the pic_df.pkl file.
        """
        # load in pkl dataframe, a pandas dataframe
        self.df = pd.read_pickle(pkl)
        self.label = label
        self.rescale = rescale
        self.preprocess_func = preprocess_func
        self.dim = dim
        self.modelname = modelname
        self.modeltype = modeltype
        # intialize these attributes as None and update as needed
        self.train_df = None
        self.test_df = None
        self.path1 = None
        self.path2 = None
        self.path3 = None
        self.path4 = None
        self.modelresult = None
        self.modelpath = None

    def create_train_test(self, test_size=0.2):
        """convert self.df into test and train pandas dataframe with specified
           classification label.
        # Arguments
           test_size: percentage of data for model testing, defaults to 0.2.
        # Returns
           test and train pandas dataframe with the specified classification
           label.
        """
        # subset the label dataframe
        sub_set = self.df[['pic_id', self.label]]
        x_train, x_test, y_train, y_test = train_test_split(
            sub_set['pic_id'], sub_set[self.label],
            stratify=sub_set[self.label], test_size=test_size
        )
        df_train = pd.concat([x_train, y_train], axis=1)
        df_test = pd.concat([x_test, y_test], axis=1)
        return df_train, df_test

    def create_generator(self, train_df, test_df, shuffle, batch_size,
                         class_mode, only_testing=False):
        """creates test and train generators for model.fit_generator.
        # Arguments
            train_df: train_df for model tuning.
            test_df: test_df for data validation tuning.
            shuffle: shuffle the data sequence for each batch generator.
            batch_size: how many images per generator.
            class_mode: None or binary in this case.
            only_testing: model testing purpose or model tuning purpose,
            default is False.
        # Returns
            generators as a tuple. If class_mode is None, creates train and
            test generators but if class_model is binary, creates train,
            classweights, and test generators.
        """
        # initialize an empty list
        generators = []
        # only generator traingenerator if it is for model tuning
        if not only_testing:
            traingen = ImageDataGenerator(
                rescale=self.rescale,
                zoom_range=[0.8, 1.7],
                shear_range=0.2,
                brightness_range=[0.5, 1.5],
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                preprocessing_function=self.preprocess_func)

            train_generator = traingen.flow_from_dataframe(
                dataframe=train_df,
                directory='../data/pics',
                x_col='pic_id',
                y_col=self.label,
                batch_size=batch_size,
                shuffle=shuffle,
                target_size=(self.dim, self.dim),
                class_mode=class_mode)
            generators.append(train_generator)
            if class_mode:
                # create classweights for train
                classweights = class_weight.compute_class_weight(
                    'balanced', np.unique(train_generator.classes),
                    train_generator.classes)
                generators.append(classweights)

        testgen = ImageDataGenerator(
            rescale=self.rescale,
            preprocessing_function=self.preprocess_func)

        test_generator = testgen.flow_from_dataframe(
            dataframe=test_df,
            directory='../data/pics',
            x_col='pic_id',
            y_col=self.label,
            batch_size=batch_size,
            shuffle=shuffle,
            target_size=(self.dim, self.dim),
            class_mode=class_mode)
        generators.append(test_generator)
        return tuple(generators)

    def save_bottleneck_features(self, file_path1, file_path2):
        """save output of model features from the vgg16 or resnet50 non dense
           layers as npy files in file_path1 and file_path2. Updates
           self.path1, self.path2, self.train and self.test df for
           model tuning.
        # Arguments
            file_path1: folder path to save train data npy.
            file_path2: folder path to save test data npy.
        """
        # intialize the model, vgg16 or ResNet50.
        # make sure not to train the top layers.
        base_model = self.modelname(weights='imagenet', include_top=False)
        # generate test_train df and updates self.train and self.test
        self.train_df, self.test_df = self.create_train_test()
        # Make sure shuffle is False so we know the label follows the sequence
        # of the dataframe so we can tune top_model and class_mode is None.
        generators = self.create_generator(self.train_df, self.test_df,
                                           shuffle=False, batch_size=16,
                                           class_mode=None)
        train_generator, test_generator = generators
        # update file_path1 and file_path2
        self.path1, self.path2 = file_path1, file_path2
        # get features saved as .npy in file_path1 and file_path2
        bottleneck_features_train = base_model.predict_generator(
            train_generator, self.train_df.shape[0]//16)
        np.save(open(self.path1, 'wb'), bottleneck_features_train)
        bottleneck_features_test = base_model.predict_generator(
            test_generator, self.test_df.shape[0]//16)
        np.save(open(self.path2, 'wb'), bottleneck_features_test)

    def resnet50_model(self, input_shape, dropout=0.25):
        """ create dense layer for resnet50 model.
        # Arguments
            input_shape: input_shape for pooling layer.
            dropout:percentage for Dropout layer to prevent overfitting,
            default is 0.25.
        # Returns
            fully connected top model for resnet50.
        """
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=input_shape))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def vgg16_model(self, input_shape, dropout=0.5):
        """ create dense layer for vgg16 model.
        # Arguments
            input_shape: input_shape for flatten layer.
            dropout:percentage for Dropout layer to prevent overfitting,
            default is 0.5.
        # Returns
            fully connected top model for vgg16.
        """
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def train_top_model(self, epoch, file_path3, verbose=0):
        """ train fully connected top model and save best weights to help fine
            tune models when unfreeze additional layers.
        # Arguments
            epoch: number of epochs in model tuning.
            file_path3: folder path to save top model best weights, h5.
            verbose: show tuning progress, default is 0
        """
        # update file_path3
        self.path3 = file_path3
        # retrieve train and test
        train_data = np.load(open(self.path1, 'rb'))
        # make sure train_data and train_label have same num of samples
        # convert string label to 1 or 0 for model tuning
        train_label = np.array(
            self.train_df[self.label].map({'0': 0, '1': 1})
        )[:-(self.train_df.shape[0] % 16)]
        test_data = np.load(open(self.path2, 'rb'))
        test_label = np.array(
            self.test_df[self.label].map({'0': 0, '1': 1})
        )[:-(self.test_df.shape[0] % 16)]
        # build top model
        if self.modeltype == 'resnet50':
            model = self.resnet50_model(train_data.shape[1:])
        if self.modeltype == 'vgg16':
            model = self.vgg16_model(train_data.shape[1:])
        model.compile(optimizer=optimizers.Adam(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        # checkpoint for best weights
        checkpoint = ModelCheckpoint(self.path3,
                                     monitor='val_acc',
                                     verbose=verbose,
                                     save_best_only=True,
                                     mode='max'
                                     )
        callbacks_list = [checkpoint]
        _, classweight, _ = self.create_generator(self.train_df, self.test_df,
                                                  False, 16, 'binary',
                                                  only_testing=False)

        model.fit(train_data, train_label,
                  epochs=epoch,
                  batch_size=16,
                  validation_data=(test_data, test_label),
                  callbacks=callbacks_list,
                  class_weight=classweight, verbose=verbose)
        # clears the model to enable next model tuning
        del model
        keras.backend.clear_session()

    def fine_tune_model(self, epoch, file_path4, verbose=0):
        """ Fine tunes the model in addition to the top model. Both vgg16 and
            resnet50 goes 4 layers up in addition to the dense layer. Updates
            self.modelresult.
        # Arguments
            epoch: number of epochs in model tuning.
            file_path4: folder path to save fine_tune model best weights, h5.
            verbose: show progress, default is 0
        """
        # build model and freeze top layers
        if self.modeltype == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False,
                                  input_shape=(self.dim, self.dim, 3))
            # build top model
            top_model = self.resnet50_model(base_model.output_shape[1:], 0.25)
        if self.modeltype == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False,
                               input_shape=(self.dim, self.dim, 3))
            top_model = self.vgg16_model(base_model.output_shape[1:], 0.5)
        # load saved weights to fine tune parameters
        top_model.load_weights(self.path3)
        # add top model to model
        model = Model(inputs=base_model.input,
                      outputs=top_model(base_model.output))
        # we will tune last 5 layers of the model for both vgg16 and resnet50
        for layer in model.layers[:-5]:
            layer.trainable = False
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=5e-5, momentum=0.9),
                      metrics=['accuracy'])
        # prepare train generator using data augmentation
        generators = self.create_generator(self.train_df, self.test_df, True,
                                           16, 'binary')
        train_generator, classweight, test_generator = generators
        # checkpoint for best weights and update file_path4
        self.path4 = file_path4
        checkpoint = ModelCheckpoint(self.path4, monitor='val_acc',
                                     verbose=verbose, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        # run and fit model
        result = model.fit_generator(
            train_generator,
            steps_per_epoch=self.train_df.shape[0]//16,
            epochs=epoch,
            validation_data=test_generator,
            validation_steps=self.test_df.shape[0]//16,
            verbose=verbose, class_weight=list(classweight),
            callbacks=callbacks_list
        )
        # update self.modelresult
        self.modelresult = result
       # clears the model to enable next model tuning
        del model
        keras.backend.clear_session()

    def plot_model_history(self):
        """plot model accuracy and loss for test and train.
        """
        # obtain model tuning history for train and test
        model_history = self.modelresult.history
        # create plots
        # subplot 1
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(model_history['val_acc'], color='red', label='test')
        plt.plot(model_history['acc'], color='blue', label='train')
        plt.title('Accuracy', size=12)
        plt.ylabel('Accuracy', size=10)
        plt.xlabel('Epoch number', size=10)
        plt.legend()
        # subplot 2
        plt.subplot(1, 2, 2)
        plt.plot(model_history['val_loss'], color='red', label='test')
        plt.plot(model_history['loss'], color='blue', label='train')
        plt.title('Loss', size=12)
        plt.ylabel('Loss', size=10)
        plt.xlabel('Epoch number', size=10)
        plt.legend()
        # show the plot
        plt.show()

    def save_model(self, modelpath):
        """save model as json. Updates selfmodelpath only once to prevent
           unncessary model saving.
        # Arguments
        modelpath: file_path to save the model as a json file
        """
        # if self.modelpath is None, update
        if not self.modelpath:
            self.modelpath = modelpath
        # check if the model is already saved
        if os.path.exists(self.modelpath):
            print('Model already exists')
            return
        model = self.modelresult.model
        model_json = model.to_json()
        # serialize model to json
        with open(self.modelpath, 'w') as json_file:
            json_file.write(model_json)

    def model_testing(self):
        """Evaluates model on the entire pic_df. Generates confusion matrix
           and prints classification_report.
        """
        # open and load the model
        json_file = open(self.modelpath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load the saved model weights
        loaded_model.load_weights(self.path4)
        # test the entire pic_df
        generators = self.create_generator(None, self.df, False, 32, 'binary',
                                           True)
        data_generator = generators[0]
        # convert labels from string to integer for model evaluation
        labels = self.df[self.label].astype('int')
        # compile models to predict the labels
        loaded_model.compile(loss='binary_crossentropy',
                             optimizer=optimizers.SGD(),
                             metrics=['accuracy']
                             )
        # predict labels
        y_pred = np.around(loaded_model.predict_generator(data_generator,
                                                          workers=8)
                           )
        # plot ROC curve with AUC value
        fpr, tpr, _ = roc_curve(labels, y_pred)
        area = roc_auc_score(labels, y_pred)
        plt.title('Receiver Operating Characteristic for '
                  f'{self.modeltype}_{self.label}')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % area)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        con_matrix = confusion_matrix(labels, y_pred)
        print(pd.DataFrame(con_matrix, index=['True_0', 'True_1'],
                           columns=['Pred_0', 'Pred_1']))
        print(classification_report(labels, y_pred))
