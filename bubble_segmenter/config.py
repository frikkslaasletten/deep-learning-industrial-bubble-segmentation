IMG_SIZE = 256
BATCH_SIZE = 10
EPOCHS_MAIN = 1000
EPOCHS_HP = 20
# If hyperparameter_optimization.py should start from scratch and overwrite existing projects of the same name
OVERWRITE = True
# If train_best_tuner.py should load existing model of the same name and resume training
LOAD = False
# If the images should be transformed to grayscale then back to RBG for compatibility
GRAY = False
# Model name for saving and loading of model weights
MODEL_NAME = "bubble_gun_model"
# Project name for saving and loading of keras tuner
PROJECT_NAME = "bubblegun_project"

# hyperparameters
BLOCK_AMOUNT_MIN = 3
BLOCK_AMOUNT_MAX = 6
BLOCK_AMOUNT_STEP = 1
BLOCK_CHANNEL_AMOUNT_MIN = 32
BLOCK_CHANNEL_AMOUNT_MAX = 64
BLOCK_CHANNEL_AMOUNT_STEP = 16
DROPOUT_MIN = 0.0
DROPOUT_MAX = 0.6
DROPOUT_STEP = 0.2