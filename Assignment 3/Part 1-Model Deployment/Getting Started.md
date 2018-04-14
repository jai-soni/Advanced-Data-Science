<h1> Gas Sensor Array Drift Dataset</h1>

<h2>About the Data</h2>

The data set is taken from UCI Machine Learning Repository, which can be accessed with below link https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset . 

The data set contains measurements from 16 chemical sensors utilized in simulations for drift compensation in a discrimination task of 6 gases at various levels of concentrations.The goal is to achieve good performance (or as low degradation as possible) over time.
The dataset was gathered within January 2007 to February 2011 (36 months) in a gas delivery platform facility situated at the ChemoSignals Laboratory in the BioCircuits Institute, University of California San Diego. Being completely operated by a fully computerized environment â€”controlled by a LabVIEW“National Instruments software on a PC fitted with the appropriate serial data acquisition boards. The measurement system platform provides versatility for obtaining the desired concentrations of the chemical substances of interest with high accuracy and in a highly reproducible manner, minimizing thereby the common mistakes caused by human intervention and making it possible to exclusively concentrate on the chemical sensors for compensating real drift.
The resulting dataset comprises recordings from six distinct pure gaseous substances, namely Ammonia, Acetaldehyde, Acetone, Ethylene, Ethanol, and Toluene, each dosed at a wide variety of concentration values ranging from 5 to 1000 ppmv.

The attributes in the data set contain set of 8 checmical features from 1 sensor with a total of 16 sensors. Thus we have a total of     16 X 8 =128 features in total

<h3> Problem Statement </h3>
<h4>Part 1:</h4>

Given the above mentioned data, Build a pipeline with Luigi that incorporates:
 1. Data ingestion into Pandas
 2. Cleaning up of Data
 3. Performing Feature Engineering , Feature Transformation and Feature selection on the data
 4. Run various  machine learning models on the data
 5. Get the Accuracy and Error metrics for all the models and store them in a csv file with ranking of models
 6. Pickle all the models and upload to S3 bucket along with error metrics

<h3> Approach</h3>
 <p><b> Data Ingestion </b><p>
  <p>1. The URL to data is provided yielding a zip file which, when unzipped contains 10 batch.dat files for different months.</p>
  <p>2. Batch files are now appended to a Pandas dataframe and saved to a CSV file with the name 'RawData.csv'.</p>
 <p><b> Data Cleaning </b></p>
 <p>1. The columns heads are a string of integers  from 0 to 128. The columns name is converted to more relevant names like 'S1F1,S1F2...S16F8', Where 'S1F2' represents Sensor:1 Feature:2 .</p>
 <p>2. From EDA, we found the data didnot contain any missing values</p>
 <p>3. The values were transformed from '90:3.399659' to 3.399659 where '90' was the column number.</p>
 <p>4. The '0' column contained the gas for which the sensor data was taken, thus column name is renamed as 'ClassNumber'.</p>
 <p>5. This data is then saved as 'CleanedData.csv' file
 <p><b> Data Preprocessing </b></p> 
 <p>1. On performing EDA, we noticed quite many outliers in the data, which would affect accuracy of the model</p>
 <p>2. After, iterating through many approaches of substituting these outliers, we finalised on substituting outlier values with median </p>
<p><b>Feature Selection</p></b>
<p>For Sensor drift data, we need to classify the data into 6 classes, i.e what data relates to what class, where class is nothing but gases numbered from 1 to 6.</p>
<p>1. For Classification problem such as this, We opted to use Sklearns make classification feature to get the feature importances.</p>
<p>2. As a result, we got 25 Important features with there ranks</p>
<p>3. These feature columns are then saved to a new CSV file with name 'SelectedData'</p>

<p><b>Classifiation Models and Predictions</p></b>
<p>1. For classification , we have divided the data into Training and Testing set with Test size of 20%.</p>
<p>2. Selected feature columns are taken as X and ClassNumber column is taken as Y</p>
<p>3. The following models are trained on this data and the Test and Train Accuracy and Error metrics are saved using pickle file.<p>
<p> Models with Test Accuracy:</p> 
 <ul>
 <li> RandomForestClassifier - 0.95
 <li> DecisionTreeClassifier - 0.90
 <li> LogisticRegression - 0.79
 <li> MLPClassifier - 0.64
 <li> GaussianNB - 0.56
</ul>
<p><b>Uploading to S3 bucket</p></b>
<p>1. The access key and Secret access keys are taken as command line arguments, the default keys are invalid</p>
<p><b> Luigi Pipeline</p></b>
<p> Luigi Pipeline provides reusability of data by providing a checkpoint type state, where if something fails , the code will start running from the previous checkpoint. This saves a lot of time since we donot have to go through all the procecss of something fails.</p>
<p>1. Five clases are made for luigi pipline which create a directed graph like structure.</p>
<p>2. Name of classes:</p>
<ul><li> download_data2csv()</li>
 <li> clean_data()</li>
 <li> feature_selection()</li>
 <li> run_models()</li>
 <li> upload2S3()</li>
 
<h2>Run the code on Docker</h2>

<h4>Pre-Requisites:</h4>
 1. Docker should be installed and running

<h4> Docker Commands to upload models in Amazon S3</h4>
 <p>1. Run the Following command and provide AWS access and secret access keys</p>
 
docker run jaisoni/gassensordata:new python Luigi_Pipeline.py upload2S3 --local-scheduler --accessKey "<AWSAccessKey>" --secretAccessKey "<AWS_SecretAccessKey>"

<h2>Run the code on local machine</h2>
<h4>Pre-Requisites:</h4>
 <p>1. Start a Ubuntu/Linux instance or system</p>
 <p>2. Clone the repository and save the Luigi_Pipeline.py in desired location</p>
 <p>3. Install python libraries give in requirements.txt file. This can be done using 'pip install -r /path/to/requirements.txt'</p>
<h4>Commands to run in Terminal</h4>
 <p>1. Run 'luigid &' to start local scheduler of luigi pipeline</p>
 <p>2. Run 'python Luigi_Pipeline.py upload2S3 --local-scheduler --accessKey "<AWSAccessKey>" --secretAccessKey "<AWS_SecretAccessKey>"'</p>
 
