<h2>Run the code on Docker</h2>
<h4>Pre-Requisites:</h4>
1. Docker should be installed and running

<h4> Docker Commands to upload models in Amazon S3</h4>
1. Run the Following command and provide AWS access and secret access keys
 
docker run jaisoni/gassensordata:new python Luigi_Pipeline.py upload2S3 --local-scheduler --accessKey "<AWSAccessKey>" --secretAccessKey "<AWS_SecretAccessKey>"
