# IMG2RECOMMENDER 
## A Social Media Marketing Business Introduction

### Business Understanding:
<img src= 'https://i2.wp.com/www.datadriveninvestor.com/wp-content/uploads/2018/11/cyber-1.png?fit=700%2C739&ssl=1' width='250'> 

* Social Media Advertisements are quite popular however:  
   
   * Paying ads on social media can be quite expensive and the click may not always lead to sales.
   * Celebrity endorsements can often lead to more sales than random ad clicks but the marketing cost is still relatively high.

This project introduces an IMG2RECOMMENDER concept for more targeted social media advertisement. This approach uses deep learning to convert social media images to more targeted product advertisements.

<img src= 'https://github.com/mzhou356/img2recommender/blob/master/image_readme/img2reconcept.png' width='350'> 


#### Problem Definition:
<img src= 'https://github.com/mzhou356/img2recommender/blob/master/image_readme/f2hbe.png' width='900'> 

* Project Goals:  

  - Successfully recommend hat, beard, or eyewear (HBE) products to customers in pictures.  
  
  - Develop a prototype program to recommend one of the 3 products to customers. 
  
#### Evaluation Criteria:

* Success Metrics:  

  - Develop successful one vs rest multi-classifiers with high AUC and high F1 scores. 
  
  - Able to recommend sensible products based upon the images seens. 

### Data Understanding and Data Preparation:

* Image Data of People: 
  
  - Manually scraped from google images 
     * total 653 usable images 
     
  - Manually labelled and cropped images of portraits for transfer learning training. 
  
  - Each image may have 2 or 3 labels. This is a multilabel classification problem. 
     * convert the folders of images into a pandas dataframe: 
     
  <img src='https://github.com/mzhou356/img2recommender/blob/master/image_readme/dataframe.png' width='300'>
 
#### Confidential Data

* All image data are webscraped from google images. Due to copyrights, they are not available on github. 

### Modeling:

* Transfer Learning:
   - VGG16: target_size: (150,150), custom vgg16 preprocess_input (see vgg16_preprocess.py for detail), 1./255 rescale. 
   - Resnet50: target_size: (224,224), default Keras ResNet preprocess_input, None rescale 
   - batch_size: 16 seems to be ideal 
   - epcho: 50 seems to be sufficient 
   - optimizers:
      * top_model_tuning: Adam() works great. 
      * fine_Model_tuning: SGD(lr = 5e-5, momentum = 0.9) 
      * unfreeze last 5 layers for both models 
      
 * Take model result averages:
    - Use prediction results from both models and average the prediction results. 
    
### Model Evaluation:

<img src='https://github.com/mzhou356/img2recommender/blob/master/image_readme/auc_roc.png' width = '1200'>

* VGG16 performed better than ResNet50: 
  - smaller dataset works better for less layers.
  - ResNet50 uses (224,224) while VGG16 uses (150,150). 
     * High resolution images may not work too well for (224,224)
       * tried ResNet50 with (150,150) but results are not promising. 
  - Hat Classifiers: 
     * vgg16 worked very well for hat. 
     * resnet50 seems to not picking up hats as well. 
  - Beard Classifiers:
    * both vgg16 and resnet50 tends to have lower sensitivity for beard than other models:
      - labelled stubs and moustaches as no beard in training data set.
          - can confuse the models 
    * both vgg16 and resnet50 tends to have lower precision for eyewear than other models:
      - sunglasses and glasses are grouped under the same category.
         - could potentially cause model confusion:
            - sunglassses: patches of dark pixels
            - glasses: outlines of dark pixels. 
    
### Deployment:

* [Blog post on the concept](https://medium.com/data-science-blogs/deep-learning-for-social-media-marketing-7c197f0c729)

#### Program Prototype
<img src= 'https://github.com/mzhou356/img2recommender/blob/master/image_readme/realexp.png' width='700'> 

* Input: an image, a folder of images, or folders of images.
   - the concept is the images would be from social media accounts and not saved locally to save space.
* load in saved models: load in 6 best vgg16 and resnet50 models before running the class ProductRecommender (see product_recommender.py for detail).
* process_1: face recognition using this [face_recognition](https://github.com/ageitgey/face_recognition) app. 
    - wrote a python class function called cropper (see product_recommender.py for detail). 
* process_2: predict all 6 labels using both models and take prediciton averages.
   * image: predict the label with the highest score.
     - If none identified, print 'no items detected'
   * folder of images or folders of images: keep a list of how many images identified as hat, beard, and eyewears and output the one with the highest percentage out of all the images.
     - keep a list of image names with the highest score for each label.
     - output image would the image name of the label chosen with the higehst score. 
   * Only recommends one product at a time to prevent overwhelming customers but store the recommendation information in the backend and update with more images uploaded on social media. 
* Output: an image with the suggested product as the image title. 
  - if just one image, print out before cropping and after cropping image as well. 
   
### Next Steps:
#### Model Improvement
* FACE2HBE:
   1. Play more with learning rate, learning rate decay, and momentum.
   2. Adjust regularization for ResNet50 such as dropout rate and batch normalization layers.
   3. Add more images especially with pictures that are falsely predicted.
   4. Create extra dense layers and/or unfreeze more layers with more images.
   5. Go from transfer learning to full CNN with more images collected and labelled.
   6. Find better face recognition software for integration.
  
#### Product Roadmap

* IMG2RECOMMENDER:
   1. Develop complex CNN models for thousands of labels.
   2. Expand products from hat, beard and eyewear items to jewelry, accessories, shoes, electronics, and more.
   3. Find various product identifier software for integration.

### Appendix:

#### File Summaries:
  * Python Files:
    - data_processing.py: helper functions to convert multilabel image folders into pandas dataframe
   for keras.preprocessing.image.ImageDataGenerator.
    - vgg16_preprocess.py: Utilities for ImageNet data preprocessing & prediction decoding. Customized the preprocess for vgg16. 
    - model_tuning_evaluation.py: model class BaseClassifier that tunes both VGG16 and ResNet50 and prints
   evaluation reports. The BaseClassifier also saves model as json and best weights as H5 files. 
      - This py file uses vgg16_preprocess.py
    - product_recommender.py: Executable file for recommmending products to social media profiles.
    
  * Jupyter Notebooks:
    - image_folders_to_dataframes.ipynb: use data_processing.py to convert multilabel image folders into pandas dataframe, checks for class imbalance, basic data information, and quickly exploratory data analysis. Saves the entire dataframe as a pickle file. 
    - vgg16_resnet50_model_tuning.ipynb: uses vgg16_preproces.py and mdoel_tuning_evaluation.py to perform model tuning, saving models and best weights, and plots model evaluation results. 
    
  * Pickle files:
      - pic_df.pkl: converted and saved by image_folders_to_dataframes.ipynb. It contains the entire 653 images with labels for all three products. 
  
  * tuning_data:
    - VGG_16_tuning:
      * vgg_model.json: model architecture for VGG16 
      * best_vgg16_model_beard.h5, best_vgg16_model_eyewear.h5, best_vgg16_model_hat.h5:
        - best final model weights for all 3 labels. 
    - resnet_data:
      * resnet50_model_5_up.json: model architecture for resnet50 
      * best_resnet50_model_beard.h5.gz, best_resnet50_model_eyewear.h5.gz, best_resnet50_model_hat.h5.gz:
        - best final model weights for all 3 labels. Saved in gz because the weight files are too large for GitHub. 
        
  * image_readme:
    - images for uploading into README.md.
    
 * img2recommender_summary_notebook.ipyn
    - a summary technical notebook for data visualization and model tuning. 
      * Checkout individual notebooks from Jupyter Notebooks for details. 
      
  * summary_presentation_img2recommender.pdf 
    - A conceptual summary of the project for non technical audiences in pdf format.
    
  * summary_presentation_img2recommender.pptx
    - A conceptual summary of the project for non technical audiences in pptx format.  
    
#### Sources: 
   * https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
   * https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
   * https://machinelearningmastery.com/check-point-deep-learning-models-keras/
   * https://machinelearningmastery.com/save-load-keras-deep-learning-models/
   * https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
   * https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
   * https://www.kaggle.com/suniliitb96/tutorial-keras-transfer-learning-with-resnet50
   * https://www.kaggle.com/pmigdal/transfer-learning-with-resnet-50-in-keras
   * https://towardsdatascience.com/transfer-learning-for-image-classification-using-keras-c47ccf09c8c8
   * https://github.com/ageitgey/face_recognition
   * https://karolakarlson.com/instagram-ads-cost-and-bidding/
   * https://www.esquire.com/lifestyle/news/a48223/celebrities-paid-ad-content-social-media/
