import os
import sys
import tensorflow as tf
from distutils.version import LooseVersion
import warnings
from platform import python_version
import shutil
from urllib.request import urlretrieve
from tqdm import tqdm
import zipfile

import Configuration


# Download progress tracker
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def verifyPythonVersion():
    assert sys.version_info >= (3, 3), "You require Python v. 3.3 or newer. You are using {a:s}".format(
        a=python_version())


# Utility to detect tensorflow version, and assert if the version is older than 1.0
def verifyTensorFlowVersion():
    assert LooseVersion(tf.__version__) >= LooseVersion(
        '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)


def issueNoGPUWarning():
    if not tf.test.gpu_device_name():
        warnings.warn(
            'No GPU found. Please use a GPU to train your neural network. Training on a CPU maybe 10x or more slower.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def isKittiDataSetAvailable():
    if (not os.path.isdir(os.path.join(Configuration.PathtoKITTIDataSet, "data_road"))):
        print(
            "Kitti data set is not available. Expected a folder 'data_road' at location {a:s}. Download from: {b:s}, and thereafter Extract the dataset in the \'KittiDataSet\' folder. This will create the subfolder \'data_road\' with all the training and test images".format(
                a=os.path.join(Configuration.PathtoKITTIDataSet, "data_road"),
                b='https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/data_road.zip'))
        sys.stdout.flush()
        assert False, "Kitti data set is not available."


# Download the VGG model if not already available locally
def downloadVGGIfNotLocallyAvailable():
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(Configuration.PathToVGGModel, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(Configuration.PathToVGGModel)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


# Check whether the augmented VGG model is availbale for use
def isAugmentedVGGModelAvailable():
    if (not os.path.isfile(os.path.join(Configuration.PathtoSaveTrainedAugmentedModel, "saved_model.pb"))):
        print(
            "Augmented VGG Model not available. Run Training first. Expected a file 'saved_model.pb' at location {a:s}".format(
                a=os.path.join(Configuration.PathtoSaveTrainedAugmentedModel, "saved_model.pb")))
        sys.stdout.flush()
        assert False, "Augmented VGG Model not available."
