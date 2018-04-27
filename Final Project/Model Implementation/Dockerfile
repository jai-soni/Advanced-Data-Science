FROM python

#Install Required libraries from requirements.txt
ADD requirements.txt /
RUN pip install -r requirements.txt

#Creating a Directory where all files will be saved
RUN mkdir "AllFiles"
RUN cd "AllFiles"

#Adding pipeline python file

ADD Sentiment_Model_Implementation_v1.py /

#Run luigi local-scheduler
CMD ["luigid","&"]

#Provide default values in case parameters are not given
#Default Amazon keys are invalid
CMD [ "python", "./Sentiment_Model_Implementation_v1.py","upload2S3","--local-scheduler","--accessKey","AKIAJA5623REBH7A","--secretAccessKey","JPtSxxGfgCWDJwlj8VGInozb82z9IJ0YHLdC"]