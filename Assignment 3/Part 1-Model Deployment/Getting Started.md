<h1> Gas Sensor Array Drift Dataset</h1>

<h2>About the Data</h2>

The data set is taken from UCI Machine Learning Repository, which can be accessed with below link https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset . 

The data set contains measurements from 16 chemical sensors utilized in simulations for drift compensation in a discrimination task of 6 gases at various levels of concentrations.The goal is to achieve good performance (or as low degradation as possible) over time.
The dataset was gathered within January 2007 to February 2011 (36 months) in a gas delivery platform facility situated at the ChemoSignals Laboratory in the BioCircuits Institute, University of California San Diego. Being completely operated by a fully computerized environment â€”controlled by a LabVIEWâ€“National Instruments software on a PC fitted with the appropriate serial data acquisition boards. The measurement system platform provides versatility for obtaining the desired concentrations of the chemical substances of interest with high accuracy and in a highly reproducible manner, minimizing thereby the common mistakes caused by human intervention and making it possible to exclusively concentrate on the chemical sensors for compensating real drift.
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
 
<h2>Run the code on Docker</h2>

<h4>Pre-Requisites:</h4>
 1. Docker should be installed and running

<h4> Docker Commands to upload models in Amazon S3</h4>
 1. Run the Following command and provide AWS access and secret access keys
 
docker run jaisoni/gassensordata:new python Luigi_Pipeline.py upload2S3 --local-scheduler --accessKey "<AWSAccessKey>" --secretAccessKey "<AWS_SecretAccessKey>"

<h2>Run the code on local machine</h2>
<h4>Pre-Requisites:</h4>
 1. Start a Ubuntu/Linux instance or system
 2. Clone the repository and save the Luigi_Pipeline.py in desired location
 3. Install python libraries give in requirements.txt file. This can be done using 'pip install -r /path/to/requirements.txt'
<h4>Commands to run in Terminal</h4>
 1. Run 'luigid &' to start local scheduler of luigi pipeline
 2. Run 'python Luigi_Pipeline.py upload2S3 --local-scheduler --accessKey "<AWSAccessKey>" --secretAccessKey "<AWS_SecretAccessKey>"'
