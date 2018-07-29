import re
import os
import random
from glob import glob
import shutil
import numpy as np
import scipy.misc
import tensorflow as tf

import Configuration


# Generator to create batches of training data
def trainingBatchGenerator(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn




# Loads the VGG model. Requires a tensorflow session to be available.
def load_vgg(sess, vgg_path):

    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    # Check if the VGG model exists
    if(not os.path.isfile(os.path.join(vgg_path, "saved_model.pb"))):
        assert False, "Required VGG Model does not exist. Abort"


    vgg_tag = ['vgg16']
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load the model
    tf.saved_model.loader.load(sess, vgg_tag, vgg_path)

    # Load the graph from VGG model
    vgg_input_tensor = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)

    # Get individual tensors for use.
    vgg_keep_prob_tensor = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor



# Augments the VGG model by adding skip layers and upscales the result to match the resolution of the input image for use
# with semantic segmentation. Only two segmentation classes are supported: "Road", and "Not Road".
def augmentVGG(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, keep_prob):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # First, ensure that all tensors are of depth 2 (= num_classes).
    vgg_layer7_out_depth2 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, padding='same',
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

    vgg_layer4_out_depth2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

    vgg_layer3_out_depth2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, padding='same',
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

    # Upsample vgg_layer7_out_depth2 by a factor 2. Kernel_regularizer is important for getting correct results.
    vgg_layer7_out_depth2_2X = tf.layers.conv2d_transpose(vgg_layer7_out_depth2, num_classes, 4, strides=(2, 2), padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

    # Fuse Layer 7 and Layer 4.
    vgg_layer7_layer4_fused = tf.add(vgg_layer7_out_depth2_2X, vgg_layer4_out_depth2)


    # Dropout
    vgg_layer7_layer4_fused = tf.layers.dropout(vgg_layer7_layer4_fused, keep_prob)


    # Upsample the result by a factor 2
    vgg_layer7_layer4_fused_2x = tf.layers.conv2d_transpose(vgg_layer7_layer4_fused, num_classes, 4, strides=(2, 2), padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))


    # Fuse the result with Layer 3
    vgg_layer7_layer4_Layer3_fused = tf.add(vgg_layer7_layer4_fused_2x, vgg_layer3_out_depth2)


    # Dropout
    vgg_layer7_layer4_Layer3_fused = tf.layers.dropout(vgg_layer7_layer4_Layer3_fused, keep_prob)


    # Upsample the result by a factor of 8.
    output = tf.layers.conv2d_transpose(vgg_layer7_layer4_Layer3_fused, num_classes, 16, strides=(8, 8), padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001), name='finalImage')

    return output



# Define optimization operations required to train the model.
def optimizeOperations(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Flatten to the number of classes being used for semantic segmentation
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # Flatten the labels too
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # Cross Entropy Loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    # Initialize an optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Return the training operation
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss



# Defines tensorflow operations to train the augmented VGG model
def trainModel(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, keepProbability):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    for thisEpoch in range(epochs):

        # We use an adaptive learning rate
        adaptiveRate = 0.001 / (1 + (thisEpoch / 10.0))

        for image, label in get_batches_fn(batch_size):
            _, loss_value = sess.run([train_op, cross_entropy_loss],
                     feed_dict={
                                    input_image: image,
                                    correct_label: label,
                                    learning_rate: adaptiveRate,
                                    keep_prob: keepProbability
                                }
                     )
        print('Epoch {a:d}: final loss {b:.2f}'.format(a=thisEpoch, b=loss_value))


def trainAugmentedVGGModel():

    # define the number of classes to be used for semantic segmentation. Currently, "Road" and "Not Road"
    numClasses = 2

    # Training Batch size. Increase if you have a more powerful GPU
    batchSize = Configuration.batchSize

    # Image size the original (i.e., download) VGG was trained for -- therefore all input images must be scaled to this size
    inputImageShape = (160, 576)

    # number of Epochs to run the training. Good results are obtained with ~20 epochs
    nEpochs = Configuration.nEpochs

    # Dropout probability during training
    TrainingDropoutProbability = 0.5

    # Placeholders that provide data to the augmented model.
    # Label from training images
    correct_label = tf.placeholder(tf.float32, [None, None, None, numClasses])

    # Learning Rate
    learning_rate = tf.placeholder(tf.float32)

    # Path to local VGG model
    pathToLocalVGGModel = os.path.join(Configuration.PathToVGGModel, "vgg")

    # Path to training images from the Kitti Road dataset
    pathToTrainingImages = os.path.join(Configuration.PathtoKITTIDataSet, "data_road/training")

    # Helper function to create batches of training data
    get_batches_fn = trainingBatchGenerator(pathToTrainingImages, inputImageShape)

    # start the tensorflow session
    with tf.Session() as sess:

        # Load the downloaded VGG model
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, pathToLocalVGGModel)

        # Augment the VGG model by adding skip layers and upscaling
        layers_output = augmentVGG(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, numClasses, keep_prob)

        # Get optimizer tensorflow operations
        logits, train_op, cross_entropy_loss = optimizeOperations(layers_output, correct_label, learning_rate, numClasses)

        # Initialize all tensorflow variables
        sess.run(tf.global_variables_initializer())

        # Train the Augmented VGG Model on the Kitti Road Dataset
        trainModel(sess,
                   nEpochs,
                   batchSize,
                   get_batches_fn,
                   train_op,
                   cross_entropy_loss,
                   input_image,
                   correct_label,
                   keep_prob,
                   learning_rate,
                   TrainingDropoutProbability)

        # Save the Model after training
        # https://stackoverflow.com/a/47235448
        if os.path.exists(Configuration.PathtoSaveTrainedAugmentedModel):
            shutil.rmtree(Configuration.PathtoSaveTrainedAugmentedModel)


        builder = tf.saved_model.builder.SavedModelBuilder(Configuration.PathtoSaveTrainedAugmentedModel)

        # Tensors to save. These will later be required for inference.
        tensor_inputImage = tf.saved_model.utils.build_tensor_info(input_image)
        tensor_logits = tf.saved_model.utils.build_tensor_info(logits)
        tensor_keep_prob = tf.saved_model.utils.build_tensor_info(keep_prob)

        # Create signature
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input_image': tensor_inputImage, 'keep_prob': tensor_keep_prob},
                outputs={'logits': tensor_logits},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    prediction_signature
            },
        )

        builder.save()
