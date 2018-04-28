#Sentiment_model_implementation :
### Running on Local when dependencies from requirements.txt are installed
python Sentiment_Model_Implementation_v1.py upload2S3 --local-scheduler --accessKey 'acc key' --secretAccessKey 'secret key'

### Run using Docker
docker run jaisoni/sentiment_model:v1 upload2S3 --local-scheduler --accessKey 'acc key' --secretAccessKey 'secret key'
