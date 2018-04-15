From python

#Add requirements.txt
ADD requirements.txt /
RUN pip install -r requirements.txt

#Create static directory

RUN cd 
RUN mkdir “static”


#Add Required files

ADD static/bootstrap.min.css /static/
ADD static/bootstrap.min.js /static/
ADD static/featureform.css /static/
ADD static/login.css /static/ 


#Create templates directory
RUN mkdir “templates”

#Add Required files

ADD templates/download.html /templates/
ADD templates/FeatureForm.html /templates/
ADD templates/FeatureFormUser.html /templates/
ADD templates/index.html /templates/
ADD templates/login.html /templates/
ADD templates/resultjson.html /templates/
ADD templates/uploadjson.html /templates/
ADD templates/welcome.html /templates/




ADD app.py /

EXPOSE 5000
CMD [ “python ”, “./app.py” ]
