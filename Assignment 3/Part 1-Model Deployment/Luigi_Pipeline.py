
# coding: utf-8

# In[1]:


#Importing  Required Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as pp
import luigi
import os
import zipfile
import urllib.request
import glob
import csv
import shutil
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#For AWS S3 Connection
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from os import listdir
from os.path import isfile, join


# In[2]:


class download_data2csv(luigi.Task):
    
    def run(self):
        try:
            #Create folders for saving zip and unzipped files
            if not os.path.exists('downloaded_zips_unzipped'):
                os.makedirs('downloaded_zips_unzipped', mode=0o777)
            else:
                shutil.rmtree(os.path.join(os.getcwd(), 'downloaded_zips_unzipped'))
                os.makedirs('downloaded_zips_unzipped', mode=0o777)
            if not os.path.exists('downloaded_zips'):
                os.makedirs('downloaded_zips', mode=0o777)
            else:
                shutil.rmtree(os.path.join(os.getcwd(), 'downloaded_zips'))
                os.makedirs('downloaded_zips', mode=0o777)
        except Exception as e:
            print(str(e))
            
        # Get Zip file from S3 Bucket and save it to downloaded zips folder
        zips = []
        zips.append(urllib.request.urlretrieve("https://s3.amazonaws.com/gassensordataset/Gassensordataset.zip", filename= 'downloaded_zips/'+'Raw_dataFile.zip'))

        zip_files = os.listdir('downloaded_zips')
        for f in zip_files:
            z = zipfile.ZipFile(os.path.join('downloaded_zips', f), 'r')
            for file in z.namelist():
                z.extract(file, r'downloaded_zips_unzipped')

        # Get the list of files in the unziped folder
        filelists = glob.glob('downloaded_zips_unzipped/Dataset' + "/*.dat")

        # Get the files in pandas DataFrame- Here we append all 10 .dat file in the same data frame
        df = pd.DataFrame()
        for file in filelists:
            if df.empty:
                df = pd.read_csv(file,sep=' ', header=None, skiprows=1)
            else:
                df = df.append(pd.read_csv(file,sep=' ', header=None, skiprows=1),ignore_index=True)
        
		#Making a CSV for the Raw dataframe
        df.to_csv(self.output().path,index = False)
        
    def output(self):
        return luigi.LocalTarget("RawData.csv")


# In[3]:

class clean_data(luigi.Task):
    
	def requires(self):
		yield download_data2csv()
    
	def run(self):
		X = pd.read_csv(download_data2csv().output().path)
		X = X[1:]
		X.reset_index(inplace=True)
		X.drop(labels='index',axis = 1, inplace =True)
		Z = X['0']
		X.drop('0', axis =1, inplace =True)
		C=0
		for i in range(1,17):
			for j in range(1,9):   
				C+=1
				C_Name = 'S'+str(i)+'F'+str(j)
				X = X.rename(columns={str(C):C_Name})
		newdf = pd.DataFrame()
		for col in X.columns.tolist():
			newdf[col] = X[col].apply(lambda X:X.split(':')[1])
		newdf['ClassNumber'] = Z
		X = newdf
		for col in X.columns.tolist():
			newdf[col]=X[col].astype('float64')
		X = newdf
		
		for i in X.columns.tolist()[:-1]:

			q75, q25 = np.percentile(X[i].dropna(), [75 ,25])
			iqr = q75 - q25

            #Find min and max values in iqr 
			min = q25 - (iqr*1.5)
			max = q75 + (iqr*1.5)
            #Get Median Value
			median = X[i].median()
            #Replace +ve and -ve outliers with median values
			X.loc[X[i] > max, i] = np.nan
			X.loc[X[i] < min, i] = np.nan
			X.fillna(median,inplace=True)
        
		# Making a CSV of this cleaned data
		X.to_csv(self.output().path,index=False)
	def output(self):
		return luigi.LocalTarget("CleanedData.csv")


# In[4]:


class feature_selection(luigi.Task):
    def requires(self):
        yield clean_data()
    
    def run(self):
        X = pd.read_csv(clean_data().output().path)
    
        #Splitting the data into training and testing set
        y = pd.DataFrame(X['ClassNumber'])
        X = X.drop(['ClassNumber'],axis=1)

        #Implementing the Random Forest Classifier for Important Feature ranking
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        X_train, y_train = make_classification(n_samples=13900, n_features=128,
                                    n_informative=2, n_redundant=0,
                                    random_state=0, shuffle=False)
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X_train, y_train)
        # Creating a list of Important features

        importance = clf.feature_importances_
        C = 0
        ivalues = []
        for i in range(1,17):
            for j in range(1,9):
                if importance[C] > 0:
                    ivalues.append('S'+str(i)+'F'+str(j))
                C+=1    
        #Storing the data of extracted features into extracted_features
        extracted_features = X[ivalues]
        
        #Adding back Target label ClassNumber to extracted_features Data Frame
        extracted_features['ClassNumber']=y['ClassNumber'].tolist()

        # Creating a file which contains Col_Name of Important features
        with open('Feature_list.txt', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in ivalues:
                writer.writerow([val])
        # Create a CSV of extracted features
        extracted_features.to_csv(self.output().path,index=False)
        
    def output(self):
        return luigi.LocalTarget("SelectedData.csv")


# In[5]:


class run_models(luigi.Task):
    def requires(self):
        yield feature_selection()
    
    def run(self):
        X = pd.read_csv(feature_selection().output().path)
        # Target label is 'ClassNumber'
        y = pd.DataFrame(X['ClassNumber'])
        
        # Removing label from dataframe
        X = X.drop(['ClassNumber'],axis=1)
        
        # Dividing data to training and testing sets with test size of 20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Opening a file for saving model accuracy 
        file = open('Model_Accuracy.txt',mode='w')
        file.write('Model'+","+'Train Accuracy'+","+'Test Accuracy') 
        file.write("\n")
        # Opening a file for saving error metrics
        file1 = open('Model_Confusion_Metrics.txt',mode='w')
        
        #creating a function which accepts parameters of training and testing data sets
        def model_Implementation(X_trn,y_trn,X_test,y_test):

            models = [                 
                        RandomForestClassifier(max_depth=8, random_state=0),
                        DecisionTreeClassifier(max_depth=8),
                        LogisticRegression(),
                        MLPClassifier(hidden_layer_sizes=(30,50,50)),
                        GaussianNB()
                      ]

            TestModels = []
            i = 0
            for model in models:
                # get model name

                # fit model on training dataset

                i = i+1
                model.fit(X_trn, y_trn)

                if i==1:
                    filename = 'RandomForestClassifier.sav'
                    TestModels.append(filename)
                elif i==2:
                    filename = 'DecisionTreeClassifier.sav'
                    TestModels.append(filename)
                elif i==3:
                    filename = 'LogisticRegression.sav'
                    TestModels.append(filename)
                elif i==4: 
                    filename = 'MLPClassifier.sav'
                    TestModels.append(filename)
                elif i==5:
                    filename = 'GaussianNB.sav'
                    TestModels.append(filename)

                pickle.dump(model, open(filename, 'wb'))

                file1.write(filename)
                file1.write("\n")

                predictions = model.predict(X_test)
                predictions_trn = model.predict(X_trn)

                accuracy_train = metrics.accuracy_score(y_train, predictions_trn)
                #print("Accuracy of the training is :", accuracy_train)

                cm = confusion_matrix(y_train, predictions_trn)
                #print(cm)
                file1.write("Training Consfusion Metrics:")
                file1.write("\n")
                file1.write(str(cm))
                file1.write("\n")

                accuracy_test = metrics.accuracy_score(y_test, predictions)
                #print("Accuracy of the testing is:", accuracy_test)

                cm = confusion_matrix(y_test, predictions)
                #print(cm)
                data = filename +","+ str(accuracy_train) +","+ str(accuracy_test)
                file.write(data) 
                file.write("\n")

                file1.write("Testing Consfusion Metrics:")
                file1.write("\n")
                file1.write(str(cm))
                file1.write("\n")
            
            file.close() 
            file1.close()
            return TestModels
            
        #Running the model
        TestModels = model_Implementation(X_train, y_train, X_test, y_test)
        TestModels.append('Model_Accuracy.txt')
        TestModels.append('Model_Confusion_Metrics.txt')
        TestModels.append('Feature_list.txt')
        model_file_names = pd.DataFrame(TestModels)
        #Writing 
        model_file_names.to_csv(self.output().path,index=False)
        
    def output(self):
        return luigi.LocalTarget("model_file_names.csv")      
        


# In[6]:


class upload2S3(luigi.Task):
    accessKey = luigi.Parameter()
    secretAccessKey = luigi.Parameter()
    def requires(self):
        yield run_models()
    def run(self):
        #To ADD
        """#Taking command line input from user
        argLen=len(sys.argv)
        accessKey=''
        secretAccessKey=''

        for i in range(1,argLen):
            val=sys.argv[i]
            if val.startswith('accessKey='):
                pos=val.index("=")
                accessKey=val[pos+1:len(val)]
                continue
            elif val.startswith('secretKey='):
                pos=val.index("=")
                secretAccessKey=val[pos+1:len(val)]
                continue"""
        
        # Get Access keys from command line 
        accessKey = self.accessKey
        secretAccessKey = self.secretAccessKey

        try:
            #Creating S3 Connection using Access and Secrect access key
            conn = S3Connection(accessKey, secretAccessKey)
            # Connecting to specified bucket
            b = conn.get_bucket('case3')
            #Initializing Key
            k = Key(b)
            #Uploading pickle and model performance files to S3 Bucket
            onlyfiles = pd.read_csv(run_models().output().path)['0'].tolist()
            for i in onlyfiles:
                k.key = i
                k.set_contents_from_filename(i)
                k.set_acl('public-read')
        except:
            print("Amazon credentials or location is invalid")


# In[7]:


if __name__=='__main__':
    luigi.run()

