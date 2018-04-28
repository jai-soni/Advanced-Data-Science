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
ADD static/main.js /static/
ADD static/main.css /static/ 


#Create templates directory
RUN mkdir “templates”

#Add Required files

ADD templates/locations.html /templates/
ADD templates/index.html /templates/
ADD templates/welcome.html /templates/


ADD mainapp.py /

EXPOSE 5000
CMD python mainapp.py