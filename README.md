## Deploying machine learning models as serverless API

This is a demo of deploying machine learning models as serverless API.

<img src="img/app_ui.png" width="700px" />

### Requirements
- AWS console access or AWS Academy Learner Lab account (running session)
- Local machine with:
- Docker
- AWS CLI v2
- A terminal (Linux/macOS/WSL)

### Setup Guide

Find detailed setup guide for deploying ML models as serverless API on AWS in article https://medium.com/@brightcode/deploying-an-ml-model-as-a-serverless-api-on-aws-5b4e9e1a0bf6   
Article contains step-by-step instructions with code samples and configurations screenshots.

### ML models

Sentiment analysis [inference_sentiment_analysis](inference_sentiment_analysis) is a modified code and ML model from AWS blog Anders Christiansen post:   
https://aws.amazon.com/blogs/machine-learning/deploying-machine-learning-models-as-serverless-apis/

Business vs Individual Classifier [inference_business_vs_individual](inference_business_vs_individual) is a modified code and ML model from 
Matthew Jones jonesmwh repository:
https://github.com/jonesmwh/business-individual-classifier

### Application architecture

<img src="img/ml-serverlsess-api-aws.png" width="700px" />

1. Web UI is served from static website hosted on Amazon S3.
2. Calling a REST API is handled with Amazon API Gateway.
3. AWS Lambda invokes a function that runs an ML model (packaged as a Docker container image in Amazon ECR)
4. Optionally request logs are stored into DynamoDB.