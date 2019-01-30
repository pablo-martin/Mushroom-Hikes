import os
import boto3

#start S3 connection
s3 = boto3.resource('s3')
#total number of files that we want to transfer
total_files = sum([len(f) for r,d,f in os.walk('flat_images/')])
print('uploading %i files to S3.' %total_files)
images_transfered = 0
for (root, dirs, files) in os.walk('flat_images/'):
    if root != 'flat_images/':
        sub_files = [root + '/' + w for w in files]
        print('Working on species %s - includes %i images' %(root, len(sub_files)))
        for sf in sub_files:
            if os.path.isfile(sf):
                data = open(sf, 'rb')
                s3.Bucket('pablo-mushroom-bucket').put_object(Key=sf, Body=data)
                data.close()
                images_transfered += 1
                if images_transfered%1 == 10000:
                    print('uploaded %i entries %i/%i' %(images_transfered, total_files))

print('Went thru %i entries, and we had %i total files' %(images_transfered,total_files))
