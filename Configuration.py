PathToVGGModel="./VGGModel"
PathtoKITTIDataSet="./KittiDataSet"
PathtoSaveTrainedAugmentedModel = "./AugmentedVGG"

# Training Batch size. Increase if you have a more powerful GPU
batchSize = 2

# number of Epochs to run the training. Good results are obtained with ~20 epochs
nEpochs = 1

# Location to keep results -- segemented images using the augmented VGG model
PathToSegmentedImages = "./SegmentedImages"

# Location of input video to segment (optional)
fullyQualifiedPathToInputMovie = "./driving.mp4"