import tensorflow as tf
import scipy.misc
import numpy as np
from threading import get_ident, local
import cv2
import ConfigurationForVideoSegmentation
def annotateFrame(inputChannel, outputChannel, pathToModel, barrier):

    # Start a tensorflow session
    allThreadData = local()

    with tf.Session() as allThreadData.sess:

        # Signature
        allThreadData.signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        # Some dictionary keys which provide access to relevant tensors in the saved model

        # Tensor which accepts input images
        allThreadData.key_input_image = 'input_image'

        # Tensor which accepts dropout probability
        allThreadData.key_keep_prob = 'keep_prob'

        # Tensor which provides the final annotated image output.
        allThreadData.key_logits = 'logits'

        # Load the graph structure
        allThreadData.meta_graph_def = tf.saved_model.loader.load(
            allThreadData.sess,
            [tf.saved_model.tag_constants.SERVING],
            pathToModel)

        # Extract the saved signature definition
        allThreadData.signature = allThreadData.meta_graph_def.signature_def

        # The name of the tensor which accepts input images -- will be used to execute "get_tensor_by_name(...)
        allThreadData.input_image_name = allThreadData.signature[allThreadData.signature_key].inputs[allThreadData.key_input_image].name

        # The name of the tensor which accepts dropout probability -- will be used to execute "get_tensor_by_name(...)
        keep_prob_name = allThreadData.signature[allThreadData.signature_key].inputs[allThreadData.key_keep_prob].name

        # The name of the tensor which provides final annotated image -- will be used to execute "get_tensor_by_name(...)
        logits_name = allThreadData.signature[allThreadData.signature_key].outputs[allThreadData.key_logits].name

        # Now get the actual tensors
        input_image = allThreadData.sess.graph.get_tensor_by_name(allThreadData.input_image_name)
        keep_prob = allThreadData.sess.graph.get_tensor_by_name(keep_prob_name)
        logits = allThreadData.sess.graph.get_tensor_by_name(logits_name)

        # dequeue a frame to annotate
        # The dequeue-annotate loop will run until there is any frame left in the inputchannel
        while(True):

            allThreadData.frame = inputChannel.get()
            allThreadData.frameNumber, allThreadData.image = zip(*allThreadData.frame.items())
            allThreadData.image = allThreadData.image[0]
            allThreadData.frameNumber = allThreadData.frameNumber[0]

            # resize the input image
            allThreadData.originalImage = np.copy(allThreadData.image)
            allThreadData.image = scipy.misc.imresize(allThreadData.image, ConfigurationForVideoSegmentation.inputResolution)


            allThreadData.clahe = cv2.createCLAHE(clipLimit=4., tileGridSize=(16, 16))
            allThreadData.lab = cv2.cvtColor(allThreadData.image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
            allThreadData.l, allThreadData.a, allThreadData.b = cv2.split(allThreadData.lab)  # split on 3 different channels
            allThreadData.l2 = allThreadData.clahe.apply(allThreadData.l)  # apply CLAHE to the L-channel
            allThreadData.lab = cv2.merge((allThreadData.l2, allThreadData.a, allThreadData.b))  # merge channels
            allThreadData.image = cv2.cvtColor(allThreadData.lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

            if(allThreadData.frameNumber == -1):
                # Break condition

                # Indicate to main thread that we are done
                if(not outputChannel.full()):

                    # Do the annotation
                    outputChannel.put({allThreadData.frameNumber: np.array(allThreadData.image)})

                    # Wait for all threads to get sentinel object
                    barrier.wait()

                    # exit the thread

                    return
                else:

                    # wait for the main thread to read all output frames and create space.
                    # When the worker hits this barrier, it serves as an indication to the master thread that the output queue is full
                    barrier.wait()

                    # The worker is allowed to cross the second barrier when the output queue has been completely processed by the main thread.
                    barrier.wait()

                break
                # return

            # Now annotate the image using the learned model

            allThreadData.im_softmax = allThreadData.sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, input_image: [allThreadData.image]})
            allThreadData.im_softmax = allThreadData.im_softmax[0][:, 1].reshape(ConfigurationForVideoSegmentation.inputResolution[0], ConfigurationForVideoSegmentation.inputResolution[1])
            allThreadData.segmentation = (allThreadData.im_softmax > 0.6).reshape(ConfigurationForVideoSegmentation.inputResolution[0], ConfigurationForVideoSegmentation.inputResolution[1], 1)
            allThreadData.mask = np.dot(allThreadData.segmentation, np.array([[0, 255, 0, 127]]))
            allThreadData.mask = scipy.misc.toimage(allThreadData.mask, mode="RGBA")
            allThreadData.street_im = scipy.misc.toimage(allThreadData.image)
            allThreadData.street_im.paste(allThreadData.mask, box=None, mask=allThreadData.mask)


            if (not outputChannel.full()):

                # Resize the image
                allThreadData.street_im = scipy.misc.imresize(np.array(allThreadData.street_im), ConfigurationForVideoSegmentation.videoResolution)

                # provide the annotated result back
                outputChannel.put({allThreadData.frameNumber: allThreadData.street_im})

            else:

                # wait for the main thread to read all output frames and create space.
                # When the worker hits this barrier, it serves as an indication to the master thread that the output queue is full
                barrier.wait()

                # The worker is allowed to cross the second barrier when the output queue has been completely processed by the main thread.
                barrier.wait()

                # Resize the image
                allThreadData.street_im = scipy.misc.imresize(np.array(allThreadData.street_im),
                                                              ConfigurationForVideoSegmentation.videoResolution)

                # provide the annotated result back
                outputChannel.put({allThreadData.frameNumber: allThreadData.street_im})
