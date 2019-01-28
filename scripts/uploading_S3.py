import os
import boto3
import pickle
import requests
import pandas as pd
import numpy as np

# Let's use Amazon S3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)

s3.Bucket('pablo-mushroom-bucket').download_file(
                            'combined_dataset_df.p', 'data.p')
df = pickle.load(open('data.p','rb'), encoding='latin1')

#creating truly unique identifier for each image
df['Unique_ID'] = \
    df['Data_Source'].apply(lambda x: x + '_') \
    + df['id'].apply(lambda x: str(x)) 


taxa_fields = ['Phylum','Class','Order','Family','Genus','Species']

def create_path_name(series):
    folders = [series.get(w).values[0] for w in taxa_fields]
    if not sum([not isinstance(w,str) for w in folders ]):
        return '/'.join(folders)
    else:
        return None
    
def generate_url_image_API_MO(image_id):
    if not np.isnan(image_id):
        url = 'https://images.mushroomobserver.org/960/' \
                                    + str(int(image_id)) + '.jpg'
        return url
    else:
        return None
    
def upload_image_S3_MO(cid, target):
    url = generate_url_image_API_MO(cid)
    r = requests.get(url, allow_redirects=True)
    open('pics/test.jpg', 'wb').write(r.content)

    # Upload a new file
    data = open('pics/test.jpg', 'rb')
    s3.Bucket('pablo-mushroom-bucket').put_object(Key=target, Body=data)
    
def upload_image_S3_DSA(Image, target):
    data = open(Image, 'rb')
    s3.Bucket('pablo-mushroom-bucket').put_object(Key=target, Body=data)
    
def create_target(Unique_ID):
    tmp = df[df.Unique_ID == Unique_ID]
    if create_path_name(tmp) != None:
        if tmp.shape[0] == 1:
            return 'pics/' + create_path_name(tmp) + '/' + Unique_ID + '.JPG'
        else:
            return ['pics/' + create_path_name(tmp) + '/' + Unique_ID + \
                       '_' + str(w+1) + '.JPG' for w in range(tmp.shape[0])]

def process_entry(Unique_ID):
    tmp = df[df.Unique_ID == Unique_ID]
    #for now we are being strict about picture having all fields
    if create_path_name(tmp) != None:
        if Unique_ID.startswith('D'):
            Image = df[df.Unique_ID == Unique_ID].Image[0]
            target = create_target(Unique_ID)
            upload_image_S3_DSA(Image, target)
        elif Unique_ID.startswith('M'):
            CIDs = tmp.Image.values
            targets = create_target(Unique_ID)
            if tmp.shape[0] > 1:
                for cid, target in zip(CIDs, targets):
                    upload_image_S3_MO(cid, target)
            else:
                upload_image_S3_MO(CIDs[0], targets)
        else:
            print('something went wrong...%s' %Unique_ID)
    else:
        return
    

J = 7338
for i, Unique_ID in enumerate(df.Unique_ID.iloc[J:]):
    if i%100 == 0:
        print('Working on entry %i/%i' %(i + J, df.Unique_ID.shape[0]))
    process_entry(Unique_ID)
