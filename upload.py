import os
import boto3
import tqdm
import sys
import subprocess

s3 = boto3.client('s3')
s3_bucket_name = 'celiaproject'
filenames = [
    # First batch of videos (done on 2023-07-05)
    # "/home/csbotos/celia_hdd/JC66 Dive 4/File 2 Dive 4 JC66_1.mov",
    # "/home/csbotos/celia_hdd/JC66 Dive 5/File 1 Dive 5 JC66_1.mov",
    # "/home/csbotos/celia_hdd/JC66 Dive 5/File 2 Dive 5 JC66_1.mov",
    # "/home/csbotos/celia_hdd/JC66 Dive 8/File 1 Dive 8 JC66_1.mov",
    # "/home/csbotos/celia_hdd/JC66 Dive 8/File 2 Dive 8 JC66_1.mov",
    # "/home/csbotos/celia_hdd/JC66 Dive 8/File 3 Dive 8 JC66_1.mov",
    # "/home/csbotos/celia_hdd/JC66 Dive 8/File 4 Dive 8 JC66_1.mov",

    # Second batch of videos (done on 2023-07-19)
    # Since they were not working on biigle
    # "/home/csbotos/celia_hdd/CH_Coral/File 5 Dive 3 JC66_1.mov",
    # "/home/csbotos/celia_hdd/JC66 Dive 2/File_5_Dive_2_JC66_1.mov",

    # Third batch of videos (done on 2023-07-20)
    # From HDD #2
    # "/home/csbotos/celia_hdd/Dive 14/File 1 Dive 14 JC66_1.mov",
    "/home/csbotos/celia_hdd/Dive 14/File 2 Dive 14 JC66_1.mov",
    "/home/csbotos/celia_hdd/Dive 14/File 3 Dive 14 JC66_1.mov",
    "/home/csbotos/celia_hdd/Dive 14/File 4 Dive 14 JC66_1.mov",
    "/home/csbotos/celia_hdd/Dive 14/File 5 Dive 14 JC66_1.mov",

    "/home/csbotos/celia_hdd/Dive 15/File 2 Dive 15 JC66_1.mov",
    "/home/csbotos/celia_hdd/Dive 15/File 3 Dive 15 JC66_1.mov",
    # "/home/csbotos/celia_hdd/Dive 15/File 4 Dive 15 JC66_1.mov",
    # "/home/csbotos/celia_hdd/Dive 15/File 5 Dive 15 JC66_1.mov",

    # "/home/csbotos/celia_hdd/Dive 16/File 1 Dive 16 JC66_1.mov",
    # "/home/csbotos/celia_hdd/Dive 16/File 2 Dive 16 JC66_1.mov",
    # "/home/csbotos/celia_hdd/Dive 16/File 3 Dive 16 JC66_1.mov",

    # "/home/csbotos/celia_hdd/Dive 17/File 1 Dive 17 JC66_1.mov",
    # "/home/csbotos/celia_hdd/Dive 17/File 2 Dive 17 JC66_1.mov",
    # "/home/csbotos/celia_hdd/Dive 17/File 3 Dive 17 JC66_1.mov",
    # "/home/csbotos/celia_hdd/Dive 17/File 4 Dive 17 JC66_1.mov",
    # "/home/csbotos/celia_hdd/Dive 17/File 5 Dive 17 JC66_1.mov",
]


temp_dir = "/home/csbotos/storage/deepsea/"

def check_format(filename):
    _format = os.path.splitext(filename)[1]
    if _format.lower() not in ['.mov', '.mp4']:
        raise ValueError(f'format {_format} not supported')
    return _format



def check_file_exists(filename):
    if not os.path.exists(filename):
        raise ValueError(f'file {filename} does not exist')
    else:
        # print file size in human readable format
        statinfo = os.stat(filename)
        print(f'file size: {statinfo.st_size / 1024 / 1024 / 1024:.2f} GB')
    

def copy_file_to_tmp(filename):
    # check if file exists and filesize is the same
    new_filename = os.path.join(temp_dir, os.path.basename(filename))
    if os.path.exists(new_filename):

        # check if file source stille exists, if not print the file size and
        # return the target filename
        if not os.path.exists(filename):
            print(f'file {filename} does not exist anymore')
            filesize = os.stat(new_filename).st_size / 1024 / 1024 / 1024
            print(f'file {new_filename} exists in temp_dir with size {filesize} GB')
            return new_filename


        src_statinfo = os.stat(filename)
        tmp_statinfo = os.stat(new_filename)
        if src_statinfo.st_size == tmp_statinfo.st_size:
            print(f'file {filename} already exists in temp_dir with same size')
            return new_filename
        else:
            print(f'file {filename} already exists in temp_dir but size is different')
            raise ValueError(f'file {filename} already exists in temp_dir but size is different')
        

    # copy file to temp_dir with rsync resume
    ret = subprocess.check_call(['rsync', '-avh', '--progress', filename, temp_dir])
    if ret != 0:
        raise ValueError(f'rsync failed to copy file {filename}')
    
    new_filename = os.path.join(temp_dir, os.path.basename(filename))
    return new_filename


def upload(fname):
    statinfo = os.stat(fname)
    with tqdm.tqdm(total=statinfo.st_size) as pbar:
        s3.upload_file(
            fname, 
            'celiaproject', 
            os.path.basename(fname), 
            ExtraArgs={'ContentType': "video/mp4"}, 
            Callback=lambda x: pbar.update(x)
        )

    bucket_location = boto3.client('s3').get_bucket_location(Bucket=s3_bucket_name)
    object_url = "https://s3.amazonaws.com/{0}/{1}".format(
        s3_bucket_name,
        os.path.basename(fname)
    )
    # object_url = "https://s3-{0}.amazonaws.com/{1}/{2}".format(
    #     bucket_location['LocationConstraint'],
    #     s3_bucket_name,
    #     os.path.basename(fname))
    
    print(f'file successfully uploaded to: {object_url}')


def convert_to_mp4(filename):
    basename = os.path.basename(filename)
    _format = check_format(filename)
    outputfile = os.path.join(temp_dir, basename.lower().replace(_format, ".mp4"))
    # replace " " with "_" in filename
    outputfile = outputfile.replace(" ", "_")
    print("target filename:", outputfile)
    # check if the subprocess call is successful
    ret = subprocess.check_call(['ffmpeg', '-i', filename, "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", outputfile])
    if ret != 0:
        raise ValueError(f'ffmpeg failed to convert file {filename}')

    return outputfile


def check_if_file_exists_in_s3(filename):
    # check if file exists in s3
    try:
        s3.head_object(Bucket=s3_bucket_name, Key=os.path.basename(filename))
        print("File found on S3!")
        return True
    except:
        print("File not found on S3")
        return False

def check_if_tmp_mp4_exists(filename):
    # check if tmp_mp4 exists
    tmp_mp4 = os.path.join(temp_dir, os.path.basename(filename).lower().replace(".mov", ".mp4"))
    
    if os.path.exists(tmp_mp4):
        print("tmp_mp4 exists")
        return tmp_mp4
    else:
        print("tmp_mp4 does not exist")
        return None

if __name__ == "__main__":
    # for filename in filenames:
    #     print("check if file exists:", filename)
    #     check_file_exists(filename)
    #     print("OK")
    
    for filename in filenames:
        # check if tmp_mp4 exists and jump to upload in that case
        tmp_mp4 = check_if_tmp_mp4_exists(filename)
        if tmp_mp4 is not None:
            print("tmp_mp4 exists, skipping to upload")    
        else:
            print("copying to temp_dir then converting to mp4")

            try:
                print(f"copy '{filename}' to temp_dir")
                tmp_mov = copy_file_to_tmp(filename)
                print("file copied to temp_dir")

                print("convert to mp4")
                tmp_mp4 = convert_to_mp4(tmp_mov)
                print("file converted to mp4:", tmp_mp4)

                print("remove .mov from temp_dir")
                os.remove(tmp_mov)
                print("file removed from temp_dir")
            except Exception as e:
                print(e)
                print("failed to convert file:", filename)
                print("skipping file:", filename)
                continue

        # check if file exists in s3 and skip upload in that case
        try:
            if check_if_file_exists_in_s3(filename):
                print("file exists in s3, skipping file:", filename)
                print("upload to s3")
            else:
                upload(tmp_mp4)
        except Exception as e:
            print(e)
            print("failed to upload file:", filename)
            print("skipping file:", filename)



