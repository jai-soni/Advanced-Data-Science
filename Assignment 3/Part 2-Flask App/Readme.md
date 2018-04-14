1) There are 2 users for our flask application, one is the admin which has privileges to access all the tuples of the predicted values after the input is been given and there is a user which has only access to one index of the predicted values.
2) The admin can give the input to the application using 3 supported ways:
	a)Upload the input file by giving the url of the file.
	b)Upload the input file from the local system.
	c)Upload the input data in the JSON Format by which the user gets results in the JSON 		format.
3) After user has successfully given the input, he will be redirected to the download results page where he can download the file of the results which are been predicted by the models use and he can compare the results of different models.
4) This way the end user gets served from our flask application by delivering the output for the data which user gives to the application.


To run the docker image of the application
The commands are as follow:
docker build -f Dockerfile .

docker tag {tag_name} username/repo:new

docker run -p 8000:5000 username/repo:new python app.py

application has been hosted on: http://ec2-52-87-245-219.compute-1.amazonaws.com

AWS Lambda Function has been created to track the training data-set updation
