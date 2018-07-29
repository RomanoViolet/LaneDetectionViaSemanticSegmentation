import skvideo.io
import ConfigurationForVideoSegmentation

# A function to extract frames from the movie to be processed.
def getNewFramesFromMovie(inputChannel, fullPathtoInputVideo):

    # Collect metadata about the video
    metaData = skvideo.io.ffprobe(fullPathtoInputVideo)

    # Number of frames
    nFrames = int(metaData['video']['@nb_frames'])

    videoGenerator = skvideo.io.vreader(fullPathtoInputVideo)

    # currentFrame counter to track progress
    currentFrame = 1

    # Extract frames
    for frame in videoGenerator:
        print("Queueing frame {a:d} of {b:d}".format(a=currentFrame, b=nFrames))

        # Each data is composed of the frame number and the frame itself -- useful for stitching back the annotated video
        inputChannel.put({currentFrame: frame})
        currentFrame = currentFrame + 1

    # Write markers (i.e., sentinel object) into the input channel to indicate the end of the video
    # Write ConfigurationForVideoSegmentation.nProcesses -1 sentinel objects
    for i in range(ConfigurationForVideoSegmentation.nProcesses -1):
        inputChannel.put({-1: frame})

    # Provider thread returns
