import urllib.request
import zipfile
import os
import pandas as pd
import logging # for logging
import shutil #to delete the directory contents
import glob

import boto.s3
import sys
from boto.s3.key import Key

import time
import datetime

#Initializing logging file #
root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch1 = logging.FileHandler('part2_log.log') #output the logs to a file
ch1.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch1.setFormatter(formatter)
root.addHandler(ch1)

ch = logging.StreamHandler(sys.stdout ) #print the logs in console as well
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

try:
    if not os.path.exists('downloaded_zips'):
        os.makedirs('downloaded_zips', mode=0o777)
    else:
        shutil.rmtree(os.path.join(os.path.dirname(__file__),'downloaded_zips'), ignore_errors=False)
        os.makedirs('downloaded_zips', mode=0o777)
    
    if not os.path.exists('downloaded_zips_unzipped'):
        os.makedirs('downloaded_zips_unzipped', mode=0o777)
    else:
        shutil.rmtree(os.path.join(os.path.dirname(__file__), 'downloaded_zips_unzipped'), ignore_errors=False)
        os.makedirs('downloaded_zips_unzipped', mode=0o777)
    logging.info('Directories cleanup complete.')
except Exception as e:
    logging.error(str(e))
    exit()     
    
    
#Reading input 
argLen=len(sys.argv)
year=sys.argv[1]
accessKey=sys.argv[2]
secretAccessKey=sys.argv[3]
inputLocation=sys.argv[4]

print("Year=",year)
print("Access Key=",accessKey)
print("Secret Access Key=",secretAccessKey)
print("Location=",inputLocation)

std_format = "http://www.sec.gov/dera/data/Public-EDGAR-log-file-data/"

link_set = []

ts = time.time()
st = datetime.datetime.fromtimestamp(ts)

############### Function to Download zips ###############
def download_zip(url):
    zips = []
    try:
        zips.append(urllib.request.urlretrieve(url, filename= 'downloaded_zips/'+url[-15:]))
        if os.path.getsize('downloaded_zips/'+url[-15:]) <= 4515: #catching empty file
            os.remove('downloaded_zips/'+url[-15:])
            logging.warning('Log file %s is empty. Attempting to download for next date.', url[-15:])
            return False
        else:
            logging.info('Log file %s successfully downloaded', url[-15:])
            return True
    except Exception as e: #Catching file not found
        logging.warning('Log %s not found...Skipping ahead!', url[-15:])
        return True
    
############### Generate URLs and download zip for the inputted year ###############

url_pre = "http://www.sec.gov/dera/data/Public-EDGAR-log-file-data/"
qtr_months = {'Qtr1':['01','02','03'], 'Qtr2':['04','05','06'], 'Qtr3':['07','08','09'], 'Qtr4':['10','11','12']}
valid_years = range(2003,2017)
days = range(1,32)

if not year:
    year = 2003
    logging.warning('Program running for 2003 by default since you did not enter any Year.')

if int(year) not in valid_years:
    logging.error("Invalid year. Please enter a valid year between 2003 and 2016.")
    exit()

logging.info('Initializing zip download.')

url_final = []
for key, val in qtr_months.items():
    for v in val:
        for d in days:
            url = url_pre +str(year) +'/' +str(key) +'/' +'log' +str(year) +str(v) + str(format(d,'02d')) +'.zip'
            if download_zip(url):
                break
            else:
                continue
logging.info('All log files downloaded for %s', year)

#Unzip the logs and extract csv #
try:
    zip_files = os.listdir('downloaded_zips')
    for f in zip_files:
        z = zipfile.ZipFile(os.path.join('downloaded_zips', f), 'r')
        for file in z.namelist():
            if file.endswith('.csv'):
                z.extract(file, r'downloaded_zips_unzipped')
    logging.info('Zip files successfully extracted to folder: downloaded_zips_unzipped.')
except Exception as e:
        logging.error(str(e))
        exit()

        

############### Load the csvs into dataframe ###############
try:
    filelists = glob.glob('downloaded_zips_unzipped' + "/*.csv")
    all_csv_df_dict = {period: pd.read_csv(period) for period in filelists}
    logging.info('All the csv read into individual dataframes')
except Exception as e:
    logging.error(str(e))
    exit()
                   
                   
#Missing Value Analysis

try:
    for key, val in all_csv_df_dict.items():
        df = all_csv_df_dict[key]
        #detecting null values
        null_count = df.isnull().sum()
        logging.info('Count of Null values for %s in all the variables:\n%s ', key, null_count)
        
        #remove rows which have no date, time, cik or accession
        df.dropna(subset=['ip'])
        df.dropna(subset=['cik'])
        df.dropna(subset=['accession'])
        df.dropna(subset=['date'])
        df.dropna(subset=['time'])
        
        # variable idx should be either 0 or 1
        incorrect_idx = (~df['idx'].isin([0.0,1.0])).sum()
        logging.info('There are %s idx which are not 0 or 1 in the log file %s', incorrect_idx, key) 
        
        # variable norefer should be either 0 or 1
        incorrect_norefer = (~df['norefer'].isin([0.0,1.0])).sum()
        logging.info('There are %s norefer which are not 0 or 1 in the log file %s', incorrect_norefer, key) 
        
        # variable noagent should be either 0 or 1
        incorrect_noagent = (~df['noagent'].isin([0.0,1.0])).sum()
        logging.info('There are %s noagent which are not 0 or 1 in the log file %s', incorrect_noagent, key) 
        
        # variable date should be same as file name
        incorrect_size = (df['size'] <= 0.0).sum()
        logging.info('There are %s rows with size less than 0 in the log file %s', incorrect_size, key) 
        
        # variable find should be either 0 or 10
        incorrect_find = (~df['find'].isin([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])).sum()
        logging.info('There are %s find which are not 0 throigh 10 in the log file %s', incorrect_find, key) 
        
        # variable crawler should be either 0 or 1
        incorrect_crawler = (~df['crawler'].isin([0.0,1.0])).sum()
        logging.info('There are %s crawler which are not 0 or 1 in the log file %s', incorrect_crawler, key) 
        
        #Recoverig nan ip with the max ip used for that cik 
        
        #for i in range(0,len(df)-1):
         #   if df.iloc[i][0] == '':
          #      tempcik = df.iloc[i][1]
           #     datacik = df[df['cik'] == tempcik]
            #    value = datacik.groupby('ip').count()
             #   max_ip = value.sort_values(['cik'],ascending=False).reset_index().iloc[0][0]
              #  df.set_value(i,'ip',max_ip)
        
        #replace nan with the most used browser in data.
        max_browser = pd.DataFrame(df.groupby('browser').size().rename('cnt')).idxmax()[0]
        df['browser'] = df['browser'].fillna(max_browser)
        
        # replace nan idx with max idx
        max_idx = pd.DataFrame(df.groupby('idx').size().rename('cnt')).idxmax()[0]
        df['idx'] = df['idx'].fillna(max_idx)
        
        # replace nan code with max code
        max_code = pd.DataFrame(df.groupby('code').size().rename('cnt')).idxmax()[0]
        df['code'] = df['code'].fillna(max_code)
        
        # replace nan norefer with zero
        df['norefer'] = df['norefer'].fillna('1')
        
        # replace nan noagent with zero
        df['noagent'] = df['noagent'].fillna('1')
        
        # replace nan find with max find
        max_find = pd.DataFrame(df.groupby('find').size().rename('cnt')).idxmax()[0]
        df['find'] = df['find'].fillna(max_find)
        
        # replace nan crawler with zero
        df['crawler'] = df['crawler'].fillna('0')
        
        # replace nan extention with max extention
        max_extention = pd.DataFrame(df.groupby('extention').size().rename('cnt')).idxmax()[0]
        df['extention'] = df['extention'].fillna(max_extention)
        
        # replace nan extention with max extention
        max_zone = pd.DataFrame(df.groupby('zone').size().rename('cnt')).idxmax()[0]
        df['zone'] = df['zone'].fillna(max_zone)
    
        # find mean of the size and replace null values with the mean
        df['size'] = df['size'].fillna(df['size'].mean(axis=0))
        
        #Recovering extention with the accession for the incorrect extention
        #for i in range(0,len(df)-1):
        #    temp = df[['accession','extention']].iloc[i][1]
        #    series = temp.split(".")    
        #    if series[0] == "":
        #        df['extention'].iloc[i] = df['accession'].iloc[i]+ df['extention'].iloc[i]
    
    
    logging.info('Rows removed where date, time, cik or accession were null.')
    logging.info('Recovered ip with the max ip used by that cik.')
    logging.info('NaN values in browser replaced with maximum count browser.')
    logging.info('NaN values in idx replaced with maximum count idx.')
    logging.info('NaN values in code replaced with maximum count code.')
    logging.info('NaN values in norefer replaced with 0.')
    logging.info('NaN values in noagent replaced with 0.')
    logging.info('NaN values in find replaced with maximum count find.')
    logging.info('NaN values in crawler replaced with 0.')
    logging.info('NaN values in extension replaced with maximum count extension.')
    logging.info('NaN values in zone replaced with maximum count zone.')
    logging.info('NaN values in size replaced with mean value of size.')
    
except Exception as e:
    logging.error(str(e))
    exit()
    
#Combining all dataframe to one csv file #
try:
    master_df = pd.concat(all_csv_df_dict)
    master_df.to_csv('master_csv.csv')
    logging.info('All dataframes of csvs are combined and exported as csv: master_csv.csv.')
except Exception as e:
    logging.error(str(e))
    print(str(e))
    exit()
    

############### Zip the csvs and logs ###############
def zipdir(path, ziph):
    ziph.write(os.path.join('master_csv.csv'))
    ziph.write(os.path.join('part2_log.log'))   

zipf = zipfile.ZipFile('Part2.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('/', zipf)
zipf.close()
logging.info('Compiled csv and log file zipped')
    

try:
    locationlist = ('us-east-2','us-east-1','us-west-1','us-west-2','ap-south-1','ap-northeast-2','ap-northeast-3','ap-southeast-1','ap-southeast-2','ca-central-1','eu-central-1','eu-west-1','eu-west-2','eu-west-3','ap-northeast-1','sa-east-1')

    loc_link = ''

    if inputLocation in locationlist:

        loc_link = 'boto.s3.connection.Location.' + inputLocation  

    print(loc_link)

    conn = boto.s3.connect_to_region(inputLocation, aws_access_key_id = accessKey,
    aws_secret_access_key = secretAccessKey)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts)    
    bucket_name = str(year) + "-"+str(st).replace(" ", "").replace("-", "").replace(":","").replace(".","")
    bucket = conn.create_bucket(bucket_name, location=inputLocation)
    print("bucket created: "+ bucket_name)
    zipfile = 'Part2.zip'
    print ("Uploading %s to Amazon S3 bucket %s", zipfile, bucket_name)

    k = Key(bucket)
    k.key = 'Part2.zip'
    k.set_contents_from_filename(zipfile)

    print('zip file uploaded')
    
except:
    logging.info("Amazon keys are invalid!!")
    print("Amazon keys are invalid!!")
    exit()