import tensorflow as tf
import scipy.misc
import numpy as np
import os
import shutil

# General settings
import Configuration


# helper function to segment a single image
def annotateImage(sess, logits, keep_prob, image_pl, image_file, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)


def segmentAllImages():

    # Image size the original (i.e., download) VGG was trained for -- therefore all input images must be scaled to this size
    inputImageShape = (160, 576)

    # Start a tensorflow session
    with tf.Session() as sess:

        # Signatures and keys useful for retrieving the tensors
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        key_input_image = 'input_image'
        key_keep_prob = 'keep_prob'
        key_logits = 'logits'

        # Load the Augmented VGG architecture
        meta_graph_def = tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            Configuration.PathtoSaveTrainedAugmentedModel)

        signature = meta_graph_def.signature_def

        # Extract the name by which tensors were saved.
        input_image_name = signature[signature_key].inputs[key_input_image].name
        keep_prob_name = signature[signature_key].inputs[key_keep_prob].name
        logits_name = signature[signature_key].outputs[key_logits].name

        # Get tensors from the saved model.
        input_image = sess.graph.get_tensor_by_name(input_image_name)
        keep_prob = sess.graph.get_tensor_by_name(keep_prob_name)
        logits = sess.graph.get_tensor_by_name(logits_name)

        # Create an output folder, if it does not exist. Delete the old folder if it exists.
        if os.path.exists(Configuration.PathToSegmentedImages):
            shutil.rmtree(Configuration.PathToSegmentedImages)

        # Create the directory
        os.makedirs(Configuration.PathToSegmentedImages)


        # Do segmentation
        for image in os.listdir(os.path.join(Configuration.PathtoKITTIDataSet, "data_road/testing/image_2")):

            annotatedImage = annotateImage(sess, logits, keep_prob, input_image, os.path.join(os.path.join(Configuration.PathtoKITTIDataSet, "data_road/testing/image_2"), image),
                                           inputImageShape)

            # save the annotated image
            scipy.misc.imsave(os.path.join(os.path.join(Configuration.PathToSegmentedImages),"out_" + image), annotatedImage)

