'''
Performs "Road" vs "No Road" segmentation over an input video.
Uses multiprocessing semantics
'''

import os

# For multiprocessing. Use multiprocessing.Process instead.
from threading import Thread

# Sorted containers
from sortedcontainers import SortedDict

# For video frame extraction and writing
import skvideo.io

# Configuration settings
import ConfigurationForVideoSegmentation

# Function that provides the frames from the movie to be processed
from GetNewFramesFromMovie import getNewFramesFromMovie

# Function that provides the design of a worker
from AnnotateFrame import annotateFrame

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
Start a thread that will extract frames and provide it for segmentation to worker threads
'''
# The thread which will feed input movie frames to annotater threads
ProviderThread = Thread(target=getNewFramesFromMovie, args=(ConfigurationForVideoSegmentation.inputChannel, ConfigurationForVideoSegmentation.fullyQualifiedPathToInputMovie))
ProviderThread.setDaemon(True)
ProviderThread.start()

# Worker threads. Each thread will annotate the frame from the movie.
allWorkerThreads = []
for thisThread in range(ConfigurationForVideoSegmentation.nProcesses -1):
    allWorkerThreads.append(Thread(target=annotateFrame, args=(ConfigurationForVideoSegmentation.inputChannel, ConfigurationForVideoSegmentation.outputChannel, ConfigurationForVideoSegmentation.PathtoSaveTrainedAugmentedModel, ConfigurationForVideoSegmentation.barrier)))
    allWorkerThreads[len(allWorkerThreads)-1].setDaemon(True)
    allWorkerThreads[len(allWorkerThreads) - 1].start()

# The main thread should deque frames from the output channel
nSentinelObjectsEncountered = 0

# Marker to exit the routine
allFramesAreDone = False

# For writing output video
fullPathtoOutputVideo = os.path.join(os.path.dirname(ConfigurationForVideoSegmentation.fullyQualifiedPathToOutputMovie), "out_" + os.path.basename(ConfigurationForVideoSegmentation.fullyQualifiedPathToInputMovie))
writer = skvideo.io.FFmpegWriter(fullPathtoOutputVideo, outputdict={'-vcodec': 'libx264', '-b': '750100000'})

while(not allFramesAreDone):

    # wait for the output channel to be full
    ConfigurationForVideoSegmentation.barrier.wait()
    # Crossing this barrier means that all workers have put their results into the output queue

    # Extract all annotated frames in increasing order. Use SortedDict for the purpose.
    s = SortedDict()

    while(not ConfigurationForVideoSegmentation.outputChannel.empty()):
        s.update(ConfigurationForVideoSegmentation.outputChannel.get())

    # Sorted container sorts by keys, and keys are frame numbers. So, we can just reverse and pop.
    for key in list(s.keys()):

        # Count the number of sentinel objects encountered
        if(key == -1):

            nSentinelObjectsEncountered = nSentinelObjectsEncountered + 1

            # Not all workers may have seen the sentinel object yet.

            if(nSentinelObjectsEncountered == ConfigurationForVideoSegmentation.nProcesses -1):
                allFramesAreDone = True
                break
        else:
            image = s[key]
            writer.writeFrame(image)
            del(s[key])


    # Let worker threads make progress
    # The logic below can be simplified at the cost of readability.
    if(allFramesAreDone == False):
        ConfigurationForVideoSegmentation.barrier.wait()
    else:
        # All worker have seen the sentinel object. Release all workers from barrier.
        ConfigurationForVideoSegmentation.barrier.wait()


 # All frames annotated
writer.close()

print("Done.")