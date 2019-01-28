

def process_entry(cid):
    Unique_ID = df.Unique_ID.iloc[cid]
    split_Unique_ID = Unique_ID.split('_')
    target = '/'.join([image_dir, df.GS_Dir.iloc[cid], Unique_ID + '.JPG'])

    def download_MO(cid):
        url = generate_url_image_API_MO(df.Image.iloc[cid])
        r = requests.get(url, allow_redirects=True)
        open(target, 'wb').write(r.content)

    if len(split_Unique_ID) == 2:
        if not os.path.isfile(target):
            if df.Data_Source.iloc[cid] == 'MO':
                download_MO(cid)
            if df.Data_Source.iloc[cid] == 'DSA':
                if not os.path.isfile(df.Image.iloc[cid]):
                    print('DSA file cannot be found: %s' %df.Image.iloc[cid])
                else:
                    os.rename(df.Image.iloc[cid], target)
    elif len(split_Unique_ID) > 2:
        assert df.Data_Source.iloc[cid] == 'MO'
        if int(split_Unique_ID[2]) == 0:
            #want to rename already downloaded image
            old_target = '_'.join(split_Unique_ID[:2])
            old_target = '/'.join([image_dir, df.GS_Dir.iloc[cid], old_target + '.JPG'])
            if os.path.isfile(old_target):
                #file already exists - just rename it
                os.rename(old_target, target)
            else:
                #actually haven't downloaded it so do so
                download_MO(cid)

        elif int(split_Unique_ID[2]) > 0:
            if not os.path.isfile(target):
                download_MO(cid)
    else:
        print('abnormal ID: %s' %Unique_ID)
