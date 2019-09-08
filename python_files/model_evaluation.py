"""helper functions to evaluate vgg16 and resnset50 models.
"""

# import libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras import optimizers
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)
from model_tuning import create_generator

def plot_model_result(model_result):
    """plot model accuracy and loss for test and train.
    # Arguments
        model_result: model_result from fine_tune_model function from
                      model_tuning.py
    """
    # obtain model tuning history for train and test
    model_history = model_result.history
    # create plots
    # subplot 1
    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    plt.plot(model_history['val_acc'], color = 'red', label = 'test')
    plt.plot(model_history['acc'], color = 'blue', label = 'train')
    plt.title('Accuracy', size=12)
    plt.ylabel('Accuracy', size=10)
    plt.xlabel('Epoch number', size=10)
    plt.legend()
    # subplot 2
    plt.subplot(1, 2, 2)
    plt.plot(model_history['val_loss'], color = 'red', label = 'test')
    plt.plot(model_history['loss'], color = 'blue', label = 'train')
    plt.title('Loss',size=12)
    plt.ylabel('Loss', size=10)
    plt.xlabel('Epoch number', size=10)
    plt.legend()
    # show the plot
    plt.show()

def save_model(model_result, file_path):
    """save model as json.
    # Arguments
        model_result: model_result from fine_tune_model function from
                      model_tuning.py
        file_path: file_path to save the model as a json file
    """
    # save the model as json
    model = model_result.model
    model_json=model.to_json()
    # serialize model to json
    with open(file_path, 'w') as json_file:
        json_file.write(model_json)


def model_testing(pkl, label, model_path, weight_path, target_size,rescale,
                  preprocess_func, model_type):
    """Evaluates model on the entire pic_df. Generates confusion matrix and
       prints classification_report.
    # Arguments
        pkl: file path to the pic_df.pkl file.
        label: eyewear, hat, or beard, a string.
        model_path: file path to the model json file from plot_save_model
        function.
        weight_path: file path to saved best model weights.
        target_size: (150,150) for vgg16 and (224,224) for resnet50.
        rescale: 1./255 for vgg16 and None for resnet50.
        preprocess_func: vgg16_preprocess input or ResNet50 preprocess input.
        model_type: resnet50 or vgg16, a string.
    """
    '''
    df: the entire picture df


    label: eyewear, hat, or beard
    '''
    df = pd.read_pickle(pkl)
    # subset the label dataframe
    sub_set = df[['pic_id',label]]
    # open and load the model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load the saved model weights
    loaded_model.load_weights(weight_path)
    # test the entire pic_df
    generators= create_generator(None, df,label,False,32,rescale,
                    preprocess_func, target_size, 'binary', True)
    data_generator = generators[0]
    # convert labels from string to integer for model evaluation
    labels = df[label].astype('int')
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
    fpr, tpr, thresholds = roc_curve(labels, y_pred)
    area = roc_auc_score(labels, y_pred)
    plt.title(f'Receiver Operating Characteristic for {model_type}_{label}')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % area)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    cm = confusion_matrix(labels, y_pred)
    print(pd.DataFrame(cm, index=['True_0','True_1'],
                       columns=['Pred_0','Pred_1'])
         )
    print(classification_report(labels, y_pred))
