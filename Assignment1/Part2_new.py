import urllib.request
import zipfile
import os
import pandas as pd
import logging # for logging
#import shutil #to delete the directory contents
import glob

import boto.s3
import sys
from boto.s3.key import Key

import time
import datetime

############### Initializing logging file ###############
root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch1 = logging.FileHandler('part2_log.log') #output the logs to a file
#ch1.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch1.setFormatter(formatter)
root.addHandler(ch1)

ch = logging.StreamHandler(sys.stdout ) #print the logs in console as well
#ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

#Reading input 
argLen=len(sys.argv)
cik=sys.argv[1]
accessionNumber=sys.argv[2]
accessKey=sys.argv[3]
secretAccessKey=sys.argv[4]
inputLocation=sys.argv[5]

#argLen=len(sys.argv)
#year = 2007
##accessionNumber=sys.argv[2]
#accessKey=''
#secretAccessKey=''
#inputLocation='us-west-2'

print("Year=",year)
print("Access Key=",accessKey)
print("Secret Access Key=",secretAccessKey)
print("Location=",inputLocation)

std_format = "http://www.sec.gov/dera/data/Public-EDGAR-log-file-data/"

#year = input("Kndly enter the year: ")

link_set = []

ts = time.time()
st = datetime.datetime.fromtimestamp(ts) 
os.makedirs('downloaded_zips_unzipped', mode=0o777)
os.makedirs('part2_zips', mode=0o777)

for i in range(1,5):

    if(i == 1):

        for j in range(1,4):

            temp = std_format + str(year) + "/" + "Qtr1" + "/" + "log" + str(year) + "0" + str(j+3*0) + "01" + ".zip"

            link_set.append(temp)

    elif(i == 2):

        for j in range(1,4):

            temp = std_format + str(year) + "/" + "Qtr2" + "/" + "log" + str(year) + "0" + str(j+3*1) + "01" + ".zip"

            link_set.append(temp)

    elif(i == 3):

        for j in range(1,4):

            temp = std_format + str(year) + "/" + "Qtr3" + "/" + "log" + str(year) + "0" + str(j+3*2) + "01" + ".zip"

            link_set.append(temp)

    else:

        for j in range(1,4):

            temp = std_format + str(year) + "/" + "Qtr4" + "/" + "log" + str(year) + str(j+3*3) + "01" + ".zip"

            link_set.append(temp)

for i in link_set:
    print(i)

zips = []

for i in link_set:
    zips.append(urllib.request.urlretrieve(i, filename= 'part2_zips/'+i[-15:]))

   
try:
    zip_files = os.listdir('part2_zips')
    for f in zip_files:
        z = zipfile.ZipFile(os.path.join('part2_zips', f), 'r')
        for file in z.namelist():
            if file.endswith('.csv'):
                z.extract(file, r'downloaded_zips_unzipped')
    logging.info('Zip files successfully extracted to folder: downloaded_zips_unzipped.')
except Exception as e:
        logging.error(str(e))
        #exit()

############### Load the csvs into dataframe ###############
    try:
        filelists = glob.glob('downloaded_zips_unzipped' + "/*.csv")
        all_csv_df_dict = {period: pd.read_csv(period) for period in filelists}
        logging.info('All the csv read into individual dataframes')
    except Exception as e:
        logging.error(str(e))
        exit()

#Missing value analysis

#consolidating the csvs to one csv

try:
    master_df = pd.concat(all_csv_df_dict)
    master_df.to_csv('master_csv.csv')
    logging.info('All dataframes of csvs are combined and exported as csv: master_csv.csv.')
except Exception as e:
    logging.error(str(e))
    exit()

############### Zip the csvs and logs ###############
def zipdir(path, ziph):
    ziph.write(os.path.join('master_csv.csv'))
#    ziph.write(os.path.join('master_df_summary.csv'))
    ziph.write(os.path.join('part2_log.log'))   

zipf = zipfile.ZipFile('Part2.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('/', zipf)
zipf.close()
logging.info('Compiled csv and log file zipped')
    


locationlist = ('us-east-2','us-east-1','us-west-1','us-west-2','ap-south-1','ap-northeast-2','ap-northeast-3','ap-southeast-1','ap-southeast-2')
#locationlist = ('APNortheast','APSoutheast','ApSoutheast2','CNNorth1','EUCentral1','EU','SAEast','USWest','USWest2')

loc_link = ''

if inputLocation in locationlist:
    
    loc_link = 'boto.s3.connection.Location.' + inputLocation  

print(loc_link)

#conn = boto.connect_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
conn = boto.s3.connect_to_region(inputLocation, aws_access_key_id = accessKey,
aws_secret_access_key = secretAccessKey)
                                 
ts = time.time()
st = datetime.datetime.fromtimestamp(ts)    
bucket_name = str(year) + "-"+str(st).replace(" ", "").replace("-", "").replace(":","").replace(".","")
bucket = conn.create_bucket(bucket_name, location=inputLocation)
print("bucket created")
zipfile = 'Part2.zip'
print ("Uploading %s to Amazon S3 bucket %s", zipfile, bucket_name)

#bucket = conn.get_bucket(bucket_name)
#k = bucket.new_key(zipfile)
k = Key(bucket)
k.key = 'Part2.zip'
k.set_contents_from_filename(zipfile)
                             
print('zip file uploaded')
#k.set_contents_from_filename(zipfile,cb=percent_cb, num_cb=10)

