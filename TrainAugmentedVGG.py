'''
Some parts of the script have been borrowed from the good guys at Udacity's Self Driving Car Nano-Degree Program.
'''


from Utils import verifyTensorFlowVersion, \
    issueNoGPUWarning, \
    verifyPythonVersion, \
    downloadVGGIfNotLocallyAvailable, \
    isKittiDataSetAvailable

from ModelBuilderUtils import trainAugmentedVGGModel

'''
Assert minimum Python version is 3.3
'''
verifyPythonVersion()


'''
Verify the minimum tensorflow version
'''
verifyTensorFlowVersion()


'''
Warn the user when the script does not find the GPU
'''
issueNoGPUWarning()


'''
Check whether Kitti dataset is available
'''
isKittiDataSetAvailable()


'''
Download VGG Model if not already available
'''
downloadVGGIfNotLocallyAvailable()


'''
Train the augmented VGG Model with Kitti Road Dataset.
The model will be trained to perform semantic segmentation "Road" vs "No Road"
for each image given.
The resulting model will also be saved locally.
'''
trainAugmentedVGGModel()
