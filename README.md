# Start

* git clone ...
* pip install - r requirements.txt


# Human age classification 

![Image alt](https://github.com/kleinar/FaceAgeClassification/raw/master/misc/group.jpg)

# Train age classification

1. Download face dataset
2. Drop the folder with the dataset in the root of the project
3. Go to the config file and set the path to the dataset.
    For example: path-to-dataset: 'UTKFace_Dataset/'
4. In the config file, write the parameters that you want to write.
    * 4.1. It is possible to choose different pre-trained neural networks: resnet18, resnet34 and so on.
    * 4.2. Resize input image for people age classifier
    * 4.3. Perform training on GPU or CPU and so on
6. Run train.py
   
# Inference

1. After training, you will receive best_model.pt in the file in which you registered in the config.
2. select the model you used as pre-trained
3. write the path to the weights of your neural network
4. Choose what you want to test your result on. On a separate image, in a folder from an image or video
5. write down the size of the input image as during training
6. Write where you want to save the results


    
Run code
* python infer.py

# Tensorboard
tensorboard --logdir=path-to-checkpoint --host=127.0.0.1
