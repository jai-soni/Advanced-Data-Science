from flask import Flask, render_template, redirect, url_for, request,jsonify
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from os import listdir
from os.path import isfile, join
from pandas.io.json import json_normalize
import sys


app = Flask(__name__)

argLen=len(sys.argv)
accessKey=''
secretAccessKey=''

for i in range(1,argLen):
    val=sys.argv[i]

    if val.startswith('accessKey='):
        pos=val.index("=")
        accessKey=val[pos+1:len(val)]
        continue
    elif val.startswith('secretAccessKey='):
        pos=val.index("=")
        secretAccessKey=val[pos+1:len(val)]
        continue

@app.route('/')
def home():
	return "Hello User, Welcome"


@app.route("/welcome")
def welcome():
	return render_template('welcome.html')


@app.route("/login", methods = ["POST", "GET"])
def login():
	error = None
	if request.method == "POST":
		if request.form['username'] == "admin" or request.form['password'] == "admin":
			return render_template("FeatureForm.html")
		elif request.form['username'] == "user" or request.form['password'] == "user":
			return render_template("FeatureFormUser.html")
		else:
			error = "Invalid credentials, please login again!"
			
	return render_template('login.html', error= error)


#@app.route("/FeatureForm", methods = ["POST","GET"])
#def FeatureForm():
	#error= None
	#if request.method == "POST":
		#if request.form['FirstFeature'] == "k":
			
			#dictt = {"FirstFeature" : request.form['FirstFeature'], "SecondFeature": request.form['SecondFeature']
			#, "ThirdFeature" : request.form['ThirdFeature'], "FourthFeature" : request.form['FourthFeature']}
			
			#Tojson = json.dumps(dictt)
			#return redirect(url_for("createJson", Tojson = Tojson))
	#return render_template('FeatureForm.html')


#@app.route("/createJson")
#def createJson():
	#return  request.args.get('Tojson')

@app.route("/uploadLocalUser",methods =["POST"])
def uploadLocalUser():
	if request.method == "POST":
		print(request.files['user_file'])
		df = pd.read_csv(request.files['user_file'])

		data = df

		resultindex = request.form['resultindex']
		resultindex = int(resultindex)

		data = df
		#Converting and cleaning the dataframe
		C = 0
		temp = []
		
		for i in range(1,17):
			for j in range(1,9):
				C+=1
				C_Name = 'S'+str(i)+'F'+str(j)
				temp.append(C_Name)
        
		data.columns = temp


		for col in data.columns.tolist():
			data[col]=data[col].astype('float64')
		with open('Feature_list.txt') as f:
			ivalues = f.read().splitlines()

		extracted_features = data[ivalues]


		#Executing Pickle file
		RFC_model = pickle.load(open("RandomForestClassifier.sav", "rb"))
		MLP_model = pickle.load(open("MLPCLassifier.sav","rb"))
		LR_model = pickle.load(open("LogisticRegression.sav","rb"))
		Gaussian_model = pickle.load(open("GaussianNB.sav","rb"))
		DTC_model = pickle.load(open("DecisionTreeClassifier.sav","rb"))


		#Using the model to predict the classes on training set
		result_RFC= RFC_model.predict(extracted_features)
		result_MLP = MLP_model.predict(extracted_features)
		result_LR = LR_model.predict(extracted_features)
		result_Gaussian = Gaussian_model.predict(extracted_features)
		result_DTC = DTC_model.predict(extracted_features)


		#Converting result of all the model predictions into dataframe 
		result_RFC = pd.DataFrame(result_RFC, columns=["RFC"])
		result_MLP = pd.DataFrame(result_MLP,columns=["MLP"])
		result_LR = pd.DataFrame(result_LR,columns=["LR"])
		result_Gaussian = pd.DataFrame(result_Gaussian, columns=["Gaussian"])
		result_DTC = pd.DataFrame(result_DTC, columns=["DTC"])



		#Concatenating all the results of model with the training set
		result_df = pd.concat([data.reset_index(drop =True), result_RFC], axis=1)
		result_df = pd.concat([result_df, result_MLP],axis=1)
		result_df = pd.concat([result_df, result_LR],axis=1)
		result_df = pd.concat([result_df, result_Gaussian],axis=1)
		result_df = pd.concat([result_df, result_DTC],axis=1)

		result_df.iloc[resultindex:resultindex+1].to_csv("result.csv", sep=",")


		#Creating S3 Connection To upload the result files
		conn = S3Connection(accessKey, secretAccessKey)
		# Connecting to specified bucket
		b = conn.get_bucket('case3')
		#Initializing Key
		k = Key(b)
		i = 'result.csv'
		k.key = i
		k.set_contents_from_filename(i)
		k.set_acl('public-read')

		end_link = "https://s3.amazonaws.com/case3/result.csv"


		table = pd.read_csv("https://s3.amazonaws.com/case3/Model_Accuracy_New.csv")

		return render_template("download.html", table= table.to_html())
	return render_template("FeatureFormUser.html")


@app.route("/uploadLocal", methods =["POST"])
def uploadLocal():
	if request.method == "POST":
		print(request.files['user_file'])
		df = pd.read_csv(request.files['user_file'])
		#y = pd.DataFrame(data['ClassNumber'])
		#X = data.drop(['ClassNumber'],axis=1)
		#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
		data = df
		#Converting and cleaning the dataframe
		C = 0
		temp = []
		
		for i in range(1,17):
			for j in range(1,9):
				C+=1
				C_Name = 'S'+str(i)+'F'+str(j)
				temp.append(C_Name)
        
		data.columns = temp


		for col in data.columns.tolist():
			data[col]=data[col].astype('float64')
		with open('Feature_list.txt') as f:
			ivalues = f.read().splitlines()

		extracted_features = data[ivalues]


		#Executing Pickle file
		RFC_model = pickle.load(open("RandomForestClassifier.sav", "rb"))
		MLP_model = pickle.load(open("MLPCLassifier.sav","rb"))
		LR_model = pickle.load(open("LogisticRegression.sav","rb"))
		Gaussian_model = pickle.load(open("GaussianNB.sav","rb"))
		DTC_model = pickle.load(open("DecisionTreeClassifier.sav","rb"))


		#Using the model to predict the classes on training set
		result_RFC= RFC_model.predict(extracted_features)
		result_MLP = MLP_model.predict(extracted_features)
		result_LR = LR_model.predict(extracted_features)
		result_Gaussian = Gaussian_model.predict(extracted_features)
		result_DTC = DTC_model.predict(extracted_features)



		#Converting result of all the model predictions into dataframe 
		result_RFC = pd.DataFrame(result_RFC, columns=["RFC"])
		result_MLP = pd.DataFrame(result_MLP,columns=["MLP"])
		result_LR = pd.DataFrame(result_LR,columns=["LR"])
		result_Gaussian = pd.DataFrame(result_Gaussian, columns=["Gaussian"])
		result_DTC = pd.DataFrame(result_DTC, columns=["DTC"])



		#Concatenating all the results of model with the training set
		result_df = pd.concat([data.reset_index(drop =True), result_RFC], axis=1)
		result_df = pd.concat([result_df, result_MLP],axis=1)
		result_df = pd.concat([result_df, result_LR],axis=1)
		result_df = pd.concat([result_df, result_Gaussian],axis=1)
		result_df = pd.concat([result_df, result_DTC],axis=1)

		result_df.to_csv("result.csv", sep=",")


		#Creating S3 Connection To upload the result files
		conn = S3Connection(accessKey, secretAccessKey)
		# Connecting to specified bucket
		b = conn.get_bucket('case3')
		#Initializing Key
		k = Key(b)
		i = 'result.csv'
		k.key = i
		k.set_contents_from_filename(i)
		k.set_acl('public-read')

		end_link = "https://s3.amazonaws.com/case3/result.csv"




		#result_df = pd.DataFrame(result)
		#result = pd.DataFrame(result).head()
		#X = X_test.iloc[:,0:6].head()
		#result_df = pd.concat([X.reset_index(drop=True),result],axis=1)

		return redirect(url_for("download"))
	return render_template("FeatureForm.html")


@app.route("/uploadLink", methods=['POST'])
def uploadLink():
	if request.method == "POST":
		link = request.form['filelink']
		df = pd.read_csv(link)		
		data = df 

		#Converting and cleaning the dataframe
		C = 0
		temp = []
		
		for i in range(1,17):
			for j in range(1,9):
				C+=1
				C_Name = 'S'+str(i)+'F'+str(j)
				temp.append(C_Name)
        
		data.columns = temp


		for col in data.columns.tolist():
			data[col]=data[col].astype('float64')
		with open('Feature_list.txt') as f:
			ivalues = f.read().splitlines()

		extracted_features = data[ivalues]

		#y = pd.DataFrame(data['ClassNumber'])
		#X = data.drop(['ClassNumber'],axis=1)

		#Splitting the data into train and test sets
		#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
		

		#Executing Pickle file
		RFC_model = pickle.load(open("RandomForestClassifier.sav", "rb"))
		MLP_model = pickle.load(open("MLPCLassifier.sav","rb"))
		LR_model = pickle.load(open("LogisticRegression.sav","rb"))
		Gaussian_model = pickle.load(open("GaussianNB.sav","rb"))
		DTC_model = pickle.load(open("DecisionTreeClassifier.sav","rb"))


		#Using the model to predict the classes on training set
		result_RFC= RFC_model.predict(extracted_features)
		result_MLP = MLP_model.predict(extracted_features)
		result_LR = LR_model.predict(extracted_features)
		result_Gaussian = Gaussian_model.predict(extracted_features)
		result_DTC = DTC_model.predict(extracted_features)


		#Converting result of all the model predictions into dataframe 
		result_RFC = pd.DataFrame(result_RFC, columns=["RFC"])
		result_MLP = pd.DataFrame(result_MLP,columns=["MLP"])
		result_LR = pd.DataFrame(result_LR,columns=["LR"])
		result_Gaussian = pd.DataFrame(result_Gaussian, columns=["Gaussian"])
		result_DTC = pd.DataFrame(result_DTC, columns=["DTC"])



		#Concatenating all the results of model with the training set
		

		result_df = pd.concat([data.reset_index(drop =True), result_RFC], axis=1)
		result_df = pd.concat([result_df, result_MLP],axis=1)
		result_df = pd.concat([result_df, result_LR],axis=1)
		result_df = pd.concat([result_df, result_Gaussian],axis=1)
		result_df = pd.concat([result_df, result_DTC],axis=1)

		result_df.to_csv("result.csv", sep=",")

		
		#Creating S3 Connection To upload the result files
		conn = S3Connection(accessKey, secretAccessKey)
		# Connecting to specified bucket
		b = conn.get_bucket('case3')
		#Initializing Key
		k = Key(b)
		i = 'result.csv'
		k.key = i
		k.set_contents_from_filename(i)
		k.set_acl('public-read')

		end_link = "https://s3.amazonaws.com/case3/result.csv"

		return redirect(url_for("download"))
	return request.form['filelink']
	


@app.route("/uploadJson")
def uploadJson():
	return render_template("uploadJson.html")


@app.route("/resultJson",methods =["POST"])
def resultJson():
	if request.method== "POST":
		#Storing the input generated by the user
		link = request.form['jsontextarea']
		print(link)

		#loading it as json
		test = json.loads(link)

		#converting the json to the Dataframe
		df = pd.DataFrame.from_dict(json_normalize(test), orient='columns')

		data = df
		#Converting and cleaning the dataframe
		C = 0
		temp = []
		
		for i in range(1,17):
			for j in range(1,9):
				C+=1
				C_Name = 'S'+str(i)+'F'+str(j)
				temp.append(C_Name)
        
		data.columns = temp


		for col in data.columns.tolist():
			data[col]=data[col].astype('float64')
		with open('Feature_list.txt') as f:
			ivalues = f.read().splitlines()

		extracted_features = data[ivalues]

		#Executing Pickle file
		RFC_model = pickle.load(open("RandomForestClassifier.sav", "rb"))
		MLP_model = pickle.load(open("MLPCLassifier.sav","rb"))
		LR_model = pickle.load(open("LogisticRegression.sav","rb"))
		Gaussian_model = pickle.load(open("GaussianNB.sav","rb"))
		DTC_model = pickle.load(open("DecisionTreeClassifier.sav","rb"))


		#Using the model to predict the classes on training set
		result_RFC= RFC_model.predict(extracted_features)
		result_MLP = MLP_model.predict(extracted_features)
		result_LR = LR_model.predict(extracted_features)
		result_Gaussian = Gaussian_model.predict(extracted_features)
		result_DTC = DTC_model.predict(extracted_features)


		#Converting result of all the model predictions into dataframe 
		result_RFC = pd.DataFrame(result_RFC, columns=["RFC"])
		result_MLP = pd.DataFrame(result_MLP,columns=["MLP"])
		result_LR = pd.DataFrame(result_LR,columns=["LR"])
		result_Gaussian = pd.DataFrame(result_Gaussian, columns=["Gaussian"])
		result_DTC = pd.DataFrame(result_DTC, columns=["DTC"])



		#Concatenating all the results of model with the training set
		result_df = pd.concat([df.reset_index(drop =True), result_RFC], axis=1)
		result_df = pd.concat([result_df, result_MLP],axis=1)
		result_df = pd.concat([result_df, result_LR],axis=1)
		result_df = pd.concat([result_df, result_Gaussian],axis=1)
		result_df = pd.concat([result_df, result_DTC],axis=1)

		#Converting Dataframe into json(String)format
		out = result_df.to_json(orient='records')

		#Writing the converted json data from dataframe into a text file
		text_file = open("Output.txt", "w")
		text_file.write(out)
		text_file.close()

		#Create a connection to S3 to upload the text file
		conn = S3Connection(accessKey, secretAccessKey)
		# Connecting to specified bucket
		b = conn.get_bucket('case3')
		#Initializing Key
		k = Key(b)
		i = 'Output.txt'
		k.key = i
		k.set_contents_from_filename(i)
		k.set_acl('public-read')


		return render_template("resultjson.html")
	
	
		

@app.route('/download')
def download():
	return render_template('download.html')

@app.route("/index")
def index():
	link =  request.args.get('df')
	return render_template("index.html", link=link)
	

@app.route("/display")
def display():
	return redirect(url_for("login"))

		
	


if __name__ == "__main__":
	app.run(debug= True,host='0.0.0.0',port = 5000)