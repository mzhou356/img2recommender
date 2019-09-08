"""Helper functions to convert multilabel image folders into pandas dataframe
   for keras.preprocessing.image.ImageDataGenerator.
"""

# import necessary libraries for image_processing
import os
import pandas as pd

def dict_from_data(file_path='/Datastore/cropped'):
    """ convert a folder with three subfolders and each subfolder name is the
        class label into a dictionary format.
        Note: this is a local folder because the image files are private.
        please modify function for your own folder structure.

    # Argument:
       file_path: a local folder path with subfolders and subfolder names are
       image labels.

    # Returns
       A dictionary with imagename as the key and the pic labels in a list
       format as the value.
    """
    # final dictionary
    result = {}
    for dirname, _, files in os.walk(file_path):
        # ignore other filess in the cropped folders, length 18 may not apply
        # in your folder structure, please modify conditions accordingly
        if len(dirname)<=18:
            continue
        # extract image label
        label = os.path.split(dirname)[-1]
        for pic in files:
            # mac issue, make sure it is not in the dictionary
            if pic == '.DS_Store':
                continue
            if pic in result:
                result[pic].append(label)
            else:
                result[pic]=[label]
    return result


def create_columns_labels(img_dict,labelnames=['eyewear','hat','beard']):
    """convert image_label dictionary from dict_from_data into a pandas
        dataframe.
    # Arguments
        img_dict: image_label dictionary from dict_from_data function
        labelnames: image labels for classification
    # Returns
        a pandas dataframe. Each label is a column with 1 as yes and 0
        as no and picture_id in sorted manner with lowest jpg number first.
    """
    # picture_id in sorted manner, ascending
    pic_id_sorted = sorted(img_dict.keys())
    # column dictionary to create dataframe
    col_dict = {}
    # initialize the pic_id column with an empty list
    col_dict['pic_id']=[]
    for pic in pic_id_sorted:
        # add pic jpg name into the pic_id column
        col_dict['pic_id'].append(pic)
        val = img_dict[pic]
        for label in labelnames:
            # initialize the key with value as an empty list
            if label not in col_dict:
                col_dict[label] = list()
            # 1 as yes
            if label in val:
                col_dict[label].append(1)
            # 0 as no
            else:
                col_dict[label].append(0)
    # convert col_dict into a pandas dataframe
    pic_df = pd.DataFrame(col_dict)
    return pic_df
