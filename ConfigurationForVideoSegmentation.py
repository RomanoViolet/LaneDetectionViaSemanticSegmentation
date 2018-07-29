# For multiprocessing
from multiprocessing import Queue, cpu_count
from threading import Barrier


# Path to the augmented VGG model trained on the Kitti Road Dataset
PathtoSaveTrainedAugmentedModel = "./AugmentedVGG"

# Location of input video to segment (optional)
fullyQualifiedPathToInputMovie = "./InputVideo/driving.mp4"

# Number of threads to use for marking. Adjust depending up the size of the graphics card
nProcesses = cpu_count() - 6

# The number of frames that cen be queued up.
inputChannel = Queue(maxsize=nProcesses*2)

# Output channel used by threads to submit annotated frames
# multiprocessing package does not yet implement priority queues. Yikes.
outputChannel = Queue(maxsize=nProcesses*2)

# Number of classes semantic segmentation model was trained for
nClasses = 2

# Resolution of each frame in the input video. (width, height, channels)
videoResolution = (720, 1280, 3)

# This is the resolution which the original VGG model accepts. Do not change.
inputResolution = (160, 576, 3)

# Barrier
# All nProcesses worker threads as well as main thread need to hit the barrier
barrier = Barrier(nProcesses)

