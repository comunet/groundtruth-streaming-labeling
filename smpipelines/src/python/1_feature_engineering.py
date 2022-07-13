"""Feature engineering for froth-anomaly-detection dataset."""
import subprocess
import sys

import argparse
import logging
import pathlib

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
install('sagemaker')

import boto3
import boto3.session

import time

import numpy as np
from PIL import Image, ImageOps
from PIL import ImageFile

# from IPython.display import display # to display images
from io import BytesIO
import os

import errno

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

ImageFile.LOAD_TRUNCATED_IMAGES = True

class S3ImagesInvalidExtension(Exception):
    pass

class S3ImagesUploadFailed(Exception):
    pass

class S3Images(object):
    """Useage:
        images = S3Images(boto_session=my_session)
        im = images.from_s3('my-example-bucket-9933668', 'pythonlogo.png')
        im
        images.to_s3(im, 'my-example-bucket-9933668', 'pythonlogo2.png')
    """
    
    def __init__(self, boto_session):
        self.s3 = boto_session.client('s3')
        

    def from_s3(self, bucket, key):
        file_byte_string = self.s3.get_object(Bucket=bucket, Key=key)['Body'].read()
        return Image.open(BytesIO(file_byte_string))
    

    def to_s3(self, img, bucket, key):
        buffer = BytesIO()
        img.save(buffer, self.__get_safe_ext(key))
        buffer.seek(0)
        sent_data = self.s3.put_object(Bucket=bucket, Key=key, Body=buffer)
        if sent_data['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise S3ImagesUploadFailed('Failed to upload image {} to bucket {}'.format(key, bucket))
        
    def __get_safe_ext(self, key):
        ext = os.path.splitext(key)[-1].strip('.').upper()
        if ext in ['JPG', 'JPEG']:
            return 'JPEG' 
        elif ext in ['PNG']:
            return 'PNG' 
        else:
            raise S3ImagesInvalidExtension('Extension is invalid')

def assert_dir_exists(path):
    """
    Checks if directory tree in path exists. If not it created them.
    :param path: the path to check if it exists
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def download_dir(client, bucket, path, target):
    """
    Downloads recursively the given S3 path to the target directory.
    :param client: S3 client to use.
    :param bucket: the name of the bucket to download from
    :param path: The S3 directory to download.
    :param target: the local directory to download the files to.
    """

    # Handle missing / at end of prefix
    if (path != "") and (not path.endswith('/')):
        path += '/'

    paginator = client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Prefix=path):
        
        if 'Contents' in result:
            # Download each file individually
            for key in result['Contents']:
                # Calculate relative path
                rel_path = key['Key'][len(path):]
                # Skip paths ending in /
                if not key['Key'].endswith('/'):
                    local_file_path = os.path.join(target, rel_path)
                    # Make sure directories exist
                    if (path != ""):
                        local_file_dir = os.path.dirname(local_file_path)
                        assert_dir_exists(local_file_dir)
                    client.download_file(bucket, key['Key'], local_file_path)
        else:
            logger.info("Nothing to download! No files found in bucket: %s, path: %s", 
                        bucket,
                        path)


def crop_image(input_image):
    #get image size
    width, height = input_image.size
    print("old size is width: {}, height: {}".format(width, height))

    #set to 512x512
    preferred_dimension_height=512
    preferred_dimension_width=512
    
    #get centre
    centre_width = width/2
    centre_height = height/2
    
    left_position = centre_width-(preferred_dimension_width/2)
    right_position = centre_width+(preferred_dimension_width/2)
    
    #our froth images are not quite central - lets get from pixel 10 at top
    top_position = 10
    bottom_position = top_position+preferred_dimension_height
    
    crop_rectangle = (left_position, top_position, right_position, bottom_position)
    cropped_im = input_image.crop(crop_rectangle)
    new_width, new_height = cropped_im.size
    print("new size is width: {}, height: {}".format(new_width, new_height))
    return cropped_im

# Pre-Process Images - Feature Engineering for Ground Truth
def preprocess_images(s3ImagesClient, s3bucketname_groundtruth_job_input, image_path, image_name):
    print("processing file: {}".format(image_name))
    raw_image = Image.open(image_path)

    # Apply a crop to make the image square for Sem Seg Algorithm
    croped_image = crop_image(raw_image)
    
    s3ImagesClient.to_s3(croped_image, s3bucketname_groundtruth_job_input, image_name)


if __name__ == "__main__":
    logger.debug("-- START feature engineering script.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-prefix", type=str, required=True)
    parser.add_argument("--s3bucketname-drop", type=str, required=True)
    parser.add_argument("--s3bucketname-groundtruth-job-input", type=str, required=True)

    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    
    localImageProcessingPath = f"{base_dir}/image_processessing"
    pathlib.Path(localImageProcessingPath).mkdir(parents=True, exist_ok=True)
    
    project_prefix = args.project_prefix
    s3bucketname_drop =args.s3bucketname_drop
    s3bucketname_groundtruth_job_input = args.s3bucketname_groundtruth_job_input
    
    # Create your own session
    my_session = boto3.session.Session()

    s3_client = boto3.client("s3", region_name="ap-southeast-2")
    s3ImagesClient = S3Images(boto_session=my_session)
    
    # download drop images
    logger.info("Downloading drop images from bucket: %s", 
                s3bucketname_drop)
    
    download_dir(s3_client, s3bucketname_drop, "", localImageProcessingPath)
    logger.debug("Drop Images downloaded.")
        
    logger.info("Loop through new drop files, process for GroundTruth")
    # Scan through local files
    for f1 in os.scandir(localImageProcessingPath):
        (filename, extension) = os.path.splitext(f1.path)
#         logger.info("filename: %s, extension: %s", filename, extension)
        if extension in [".png", ".jpg", ".jpeg"]:
            logger.info("file: %s", f1.name)
            camera_name = f1.name.split(' ')[0]

            # Do main activity - prepare image and put in GroundTruth INPUT bucket
            preprocess_images(s3ImagesClient, s3bucketname_groundtruth_job_input, f1.path, f1.name)
            
            # now delete s3 file from drop
            source_key = "{}".format(f1.name)
            s3_client.delete_object(Bucket = s3bucketname_drop, Key = source_key)
            logger.info("Processed file: {}".format(source_key))
            # ...
    logger.info("Files processed. Kick start chained GroundTruth job.")
    
    

    logger.info("--END feature engineering script --")
