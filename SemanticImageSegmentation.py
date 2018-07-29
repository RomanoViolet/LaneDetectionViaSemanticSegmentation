'''
Some parts of the script have been borrowed from the good guys at Udacity's Self Driving Car Nano-Degree Program.
'''


from Utils import verifyTensorFlowVersion, \
    issueNoGPUWarning, \
    verifyPythonVersion, \
    isAugmentedVGGModelAvailable, \
    isKittiDataSetAvailable

from InferenceUtils import segmentAllImages

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
Check whether Augmented VGG Model is available.
'''
isAugmentedVGGModelAvailable()


'''
Check whether Kitti dataset is available
'''
isKittiDataSetAvailable()



'''
Do Segmentation "Road" vs "Not Road" using the augmented VGG Model.
The inputs are test images from the Kitti Road DataSet.
Images are saved in $(Configuration.PathToSegmentedImages)
'''
segmentAllImages()