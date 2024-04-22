This CNN model classifies images with a 92.03% accuracy rate.

Model Architecture:
Layers: 3 convolutional layers with max-pooling; dropout layers to reduce overfitting.
Filters: 32, 64, and 128 in successive layers.

Data Handling:
Image normalization and dimension transposition.
One-hot encoding for labels.
Split into training and validation sets.

Training Details:
Utilizes ADAM optimizer.
Visualizations for training/validation loss and accuracy are provided to guide adjustments.
Potential Enhancements:

Explore different optimizers and learning rate adjustments for improved performance.