
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
from bs4 import BeautifulSoup
import urllib3
import sys
import requests
import os
import zipfile
import logging
import time
import datetime
import boto
from boto.s3.key import Key


# In[2]:


#Log file initialization
root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch1 = logging.FileHandler('Log_P1.log') #output the logs to a file
ch1.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch1.setFormatter(formatter)
root.addHandler(ch1)

ch = logging.StreamHandler(sys.stdout ) #print the logs in console as well
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


# In[3]:


#Taking command line input from user
argLen=len(sys.argv)
cik=''
accNumber=''
accessKey=''
secretAccessKey=''
inputLocation=''

for i in range(1,argLen):
    val=sys.argv[i]
    if val.startswith('cik='):
        pos=val.index("=")
        cik=val[pos+1:len(val)]
        continue
    elif val.startswith('accNumber='):
        pos=val.index("=")
        accNumber=val[pos+1:len(val)]
        continue
    elif val.startswith('accessKey='):
        pos=val.index("=")
        accessKey=val[pos+1:len(val)]
        continue
    elif val.startswith('secretKey='):
        pos=val.index("=")
        secretAccessKey=val[pos+1:len(val)]
        continue
    elif val.startswith('location='):
        pos=val.index("=")
        inputLocation=val[pos+1:len(val)]
        continue
        
'''argLen=len(sys.argv)
cik=sys.argv[1]
accessionNumber=sys.argv[2]
accessKey=sys.argv[3]
secretAccessKey=sys.argv[4]
inputLocation=sys.argv[5]'''
        
#CIK and Accession Number not provided, Setting to Default value
if len(cik)==0:
    cik='51143'
if len(accNumber)==0:
    accNumber='000005114313000007'
    
print("CIK=",cik)
print("Accession Number=",accNumber)
print("Access Key=",accessKey)
print("Secret Access Key=",secretAccessKey)
print("Location=",inputLocation)

#Creating URL using CIK and Accession number 
link1 = "http://www.sec.gov/Archives/edgar/data/" + cik + "/"+accNumber +"/"+ accNumber[:10]+ "-"+accNumber[10:12]+"-"+accNumber[12:]+ "-"+"index.html"
request = requests.get(link1)

if request.status_code == 200:
    print('Web site exists')
else:
    print('Web site does not exist, Invalid CIK or Accession Number',request.status_code)
    exit()


# In[4]:


#function to zip a file
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


# In[6]:


#Filtering , Pre-processing and Zipping the tables in 10-Q
http = urllib3.PoolManager()
response = http.request('GET', link1)
doc = BeautifulSoup(response.data)

tables = doc.findAll('table')
for table in tables:
    #find th = 'Type' in the selected table and its index.
    done = 'false'
    rows = table.findAll('tr')
    links = table.findAll('a')
    for row in rows:
        rows_th = row.findAll('th')
        rows_td = row.findAll('td')
        for row in rows_th:
            if row.text.lower() == 'type':
                type_ind = rows_th.index(row)
        
        col = [row.text for row in rows_td]
        type_col = col[type_ind::type_ind]
        
        for item in type_col:
            if str(item) == '10-Q':#Checking If 10-Q is found in any of the tables in the link
                aim = type_col.index(item)*type_ind
                for link in links:
                    if links.index(link) == aim:
                        final_link = "https://www.sec.gov"+ str(link).split('<a href="')[1].split('"')[0]
                        table_list = pd.read_html(final_link)
                        print(final_link)
                        http = urllib3.PoolManager()
                        response = http.request('GET', final_link)
                        doc = BeautifulSoup(response.data)

                        tables = doc.findAll('table')
                        
                        #Filtering the required tables based on Background colour
                        Required_Indexes = []

                        for j in range(0,len(tables)):
                            rows = tables[j].findAll('td')
                            for i in range(0,len(rows)):
                                if 'background:' in str(rows[i]):
                                    Required_Indexes.append(j)
                                    break
                        for k in Required_Indexes:
                            row_start_value=0
                            for rows in range(0,len(table_list[k])-1):
                                if table_list[k][rows:rows+1][0].notnull()[rows]:
                                    row_start_value=rows
                                    break
                            s=[]
                            for j in range(row_start_value,len(table_list[k])-1):
                                s.append(table_list[k].iloc[j].dropna().tolist())
                                for i in range(0,len(s)):
                                    if '$' in s[i]:
                                        s[i].remove('$') 
                            a= pd.DataFrame(s)
                            #Saving the file to current woring directory
                            cwd = os.getcwd()
                            if not os.path.exists(cwd+"/"+accNumber):
                                os.makedirs(cwd+"/"+accNumber)
                            a.to_csv(cwd+"/"+accNumber+"/Table "+str(k+1)+ ".csv", sep=',', header= False,index= False)
                        time.sleep(30)
                        if __name__ == '__main__':
                            cwd = os.getcwd()
                            zipf = zipfile.ZipFile('%s.zip' %cik, 'w', zipfile.ZIP_DEFLATED)
                            zipf.write(os.path.join('Log_P1.log'))
                            zipdir(accNumber, zipf)
                            zipf.close()


# In[15]:


locationlist = ('us-east-2','us-east-1','us-west-1','us-west-2','ap-south-1','ap-northeast-2','ap-northeast-3','ap-southeast-1','ap-southeast-2')
#locationlist = ('APNortheast','APSoutheast','ApSoutheast2','CNNorth1','EUCentral1','EU','SAEast','USWest','USWest2')

loc_link = ''

if inputLocation in locationlist:
    
    loc_link = 'boto.s3.connection.Location.' + inputLocation  

#print(loc_link)
#try:
#conn = boto.connect_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
    conn = boto.s3.connect_to_region(inputLocation, aws_access_key_id = accessKey,
    aws_secret_access_key = secretAccessKey)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts)    
    bucket_name = accNumber.lower()+ "-"+str(st).replace(" ", "").replace("-", "").replace(":","").replace(".","")
    bucket = conn.create_bucket(bucket_name, location=inputLocation)
    print("bucket created")
    zipfile = zipf.filename
    print ("Uploading %s to Amazon S3 bucket %s", zipfile, bucket_name)

    #bucket = conn.get_bucket(bucket_name)
    #k = bucket.new_key(zipfile)
    k = Key(bucket)
    k.key = zipf.filename

    def percent_cb(complete, total):
            sys.stdout.write('.')
            sys.stdout.flush()

    k.set_contents_from_filename(zipfile)

    print('zip file uploaded')
    k.set_contents_from_filename(zipfile)

#except:
    #logging.info("Amazon credentials or location is invalid")
    #print("Amazon credentials or location is invalid")
    #exit()


# In[24]:


#Validating Amazon Keys
if not accessKey or not secretAccessKey:
    logging.warning('Access Key and Secret Access Key not provided!!')
    print('Access Key and Secret Access Key not provided!!')
    exit()

AWS_ACCESS_KEY_ID = accessKey
AWS_SECRET_ACCESS_KEY = secretAccessKey

try:
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
    print("Connected to S3")

except:
    logging.info("Amazon keys are invalid!!")
    print("Amazon keys are invalid!!")
    exit()
    

