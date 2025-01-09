# Machine_Learing-For_Vehicle_Detection_in-Diverse_Weather_Condition

## Project Description

This project aims to classify weather images into different categories, such as sunny, cloudy, rainy, foggy, snowy, etc., using the EfficientNetB0 model. We leverage transfer learning by utilizing the pre-trained weights of EfficientNetB0 on the ImageNet dataset and fine-tuning it for our specific weather image classification task. 

The project demonstrates how to build and train a deep learning model for image classification, evaluate its performance, and visualize the model's decision-making process using Grad-CAM.

## Dataset

The dataset used for this project is "juvdv2-vdvwc", which contains images of various weather conditions. The dataset is structured into separate folders for training and validation, with each folder containing subfolders for each weather category.

**Dataset Structure:**
juvdv2-vdvwc/ ├── Train/ │ ├── Class1/ (e.g., Sunny) │ │ ├── image1.jpg │ │ ├── image2.jpg │ │ └── ... │ ├── Class2/ (e.g., Cloudy) │ │ ├── image1.jpg │ │ ├── image2.jpg │ │ └── ... │ └── ... └── Val/ ├── Class1/ (e.g., Sunny) │ ├── image1.jpg │ ├── image2.jpg │ │ └── ... ├── Class2/ (e.g., Cloudy) │ ├── image1.jpg │ ├── image2.jpg │ └── ... └── ...
**Data Preprocessing:**

* **Image Resizing:** All images are resized to 224x224 pixels.
* **Normalization:** Pixel values are scaled to the range [0, 1] by dividing by 255.
* **Data Augmentation:** The training data is augmented using `ImageDataGenerator` to increase the diversity of the training set and improve model generalization. Augmentation techniques include:
    * Random shearing
    * Random zooming
    * Horizontal flipping
    * Random brightness adjustments
    * Random rotations
    * Random width and height shifts


## Model Architecture

The model is built using the EfficientNetB0 architecture as the base model, followed by custom layers for classification.

**Base Model:**

* EfficientNetB0 pre-trained on ImageNet is used as the base model.
* The top classification layers of EfficientNetB0 are removed (`include_top=False`) to allow for custom classification layers.

**Custom Layers:**

* **Flatten:** Flattens the output of the base model to a 1D vector.
* **Dense (128 units, ReLU activation):** A fully connected layer with 128 units and ReLU activation function.
* **Dropout (0.5):** A dropout layer with a rate of 0.5 to prevent overfitting.
* **Dense (num_classes, softmax activation):** The final classification layer with the number of units equal to the number of weather categories and softmax activation function to output probabilities for each class.

## Training

The model is trained using the following settings:

* **Optimizer:** Adam optimizer with a learning rate of 1e-4.
* **Loss Function:** Categorical cross-entropy loss function, suitable for multi-class classification.
* **Metrics:** Accuracy, Precision, Recall, AUC (Area Under the Curve), and F1-score are used to monitor the model's performance during training and evaluation.
* **Callbacks:**
    * **EarlyStopping:** Stops training if the validation loss does not improve for a certain number of epochs (patience=10), preventing overfitting.
    * **ModelCheckpoint:** Saves the best model based on the validation loss during training.
    * **ReduceLROnPlateau:** Reduces the learning rate if the validation loss plateaus, helping the model converge better.
    * **F1ScoreCallback:** A custom callback to calculate and print the F1-score at the end of each epoch.
* **Epochs:** The model is trained for 20 epochs.

## Evaluation

The trained model is evaluated on the validation set using the following metrics:

* **Accuracy:** The overall accuracy of the model in correctly classifying weather images.
* **Precision:** The proportion of correctly predicted positive instances out of all predicted positive instances.
* **Recall:** The proportion of correctly predicted positive instances out of all actual positive instances.
* **F1-score:** The harmonic mean of precision and recall, providing a balanced measure of the model's performance.
* **AUC:** The area under the receiver operating characteristic curve, measuring the model's ability to distinguish between different classes.
* **mAP (mean Average Precision):** The average precision score across all classes, providing a comprehensive measure of the model's performance.
* **Confusion Matrix:** A visualization of the model's performance by showing the counts of true positive, true negative, false positive, and false negative predictions for each class.

## Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) is employed to visualize the regions of the input image that are most influential in the model's prediction.

* **`get_gradcam_heatmap` function:** Calculates the Grad-CAM heatmap by computing the gradients of the target class with respect to the feature maps of the last convolutional layer.
* **`display_gradcam` function:** Overlays the heatmap on the original image to highlight the important regions.


## Usage

1. **Clone the repository:** `!git clone https://github.com/Sourajit-Maity/juvdv2-vdvwc.git`
2. **Install the necessary libraries:** `!pip install tensorflow efficientnet opencv-python matplotlib seaborn scikit-learn`
3. **Run the notebook cells:** Execute the cells in the provided Jupyter Notebook to load the data, define the model, train it, evaluate its performance, and visualize the results using Grad-CAM.


## Results

The results of the model evaluation, including accuracy, precision, recall, F1-score, AUC, and mAP, are presented in the notebook. The confusion matrix and Grad-CAM visualizations provide further insights into the model's performance and areas for improvement.


## Conclusion

This project demonstrates the effectiveness of transfer learning with EfficientNetB0 for weather image classification. The model achieves promising results in accurately classifying different weather conditions.

**Future Work:**

* Explore using larger and more diverse datasets for training.
* Experiment with different EfficientNet variants (B1, B2, etc.) or other architectures.
* Fine-tune hyperparameters to optimize model performance.
* Implement ensemble methods to combine predictions from multiple models.
* Develop a web application or mobile app to deploy the model for real-world use.
