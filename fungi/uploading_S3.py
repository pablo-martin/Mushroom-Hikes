import os
import boto3
import botocore
import logging

logging.getLogger('boto').setLevel(logging.CRITICAL)

logger = logging.getLogger()
# start S3 connection
s3 = boto3.resource('s3')

# check if file exists in S3


def check_if_file_exists(sf):
    try:
        s3.Object('pablo-mushroom-bucket', sf).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise
    else:
        return True


def upload_image(sf):
    try:
        with open(sf, 'rb') as data:
            s3.Bucket('pablo-mushroom-bucket').put_object(Key=sf, Body=data)
    except botocore.exceptions.ClientError as e:
        logging.error(e)


def upload_images(PARENT_DIR='flat_images/'):

    # total number of files that we want to transfer
    total_files = sum([len(f) for r, d, f in os.walk(PARENT_DIR)])
    print('uploading %i files to S3.' % total_files)
    images_transfered = 0
    for (root, dirs, files) in os.walk(PARENT_DIR):
        if root != PARENT_DIR:
            sub_files = [root + '/' + w for w in files]
            print('Working on species %s - includes %i images' % (root, len(sub_files)))
            for sf in sub_files:
                if os.path.isfile(sf) and not check_if_file_exists(sf):
                    upload_image(sf)

                    images_transfered += 1
                    if images_transfered % 1 == 10000:
                        print('uploaded %i entries %i/%i' % (images_transfered, total_files))

    print('Went thru %i entries, and we had %i total files' % (images_transfered, total_files))
