#!/bin/bash

# Synapty AWS Lambda Deployment Script
# Usage: ./deploy-lambda.sh [function-name] [openai-api-key]

set -e

FUNCTION_NAME=${1:-synapty-backend}
OPENAI_API_KEY=${2}
AWS_ACCOUNT_ID="653858369289"
AWS_REGION=$(aws configure get region || echo "us-east-1")

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Usage: ./deploy-lambda.sh [function-name] [openai-api-key]"
    echo "Example: ./deploy-lambda.sh synapty-backend sk-proj-..."
    exit 1
fi

echo "🎯 Deploying to AWS Account: $AWS_ACCOUNT_ID"
echo "🌍 Using AWS Region: $AWS_REGION"

echo "🚀 Deploying Synapty backend to AWS Lambda..."

# Create deployment directory
echo "📦 Creating deployment package..."
rm -rf lambda-deploy
mkdir lambda-deploy
cd lambda-deploy

# Copy the unified simple_main.py
echo "📝 Copying Lambda function code..."
cp ../backend/simple_main.py .

# Install requests library
echo "📥 Installing dependencies..."
pip install requests -t . --quiet

# Remove unnecessary files
rm -rf __pycache__ *.pyc

# Create deployment package
echo "🗜️  Creating ZIP package..."
zip -r synapty-backend.zip . -q

# Check if Lambda function exists
echo "🔍 Checking if Lambda function exists..."
if aws lambda get-function --function-name $FUNCTION_NAME > /dev/null 2>&1; then
    echo "📝 Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://synapty-backend.zip
    
    # Update environment variables
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --environment Variables="{OPENAI_API_KEY=$OPENAI_API_KEY}"
else
    echo "🆕 Creating new Lambda function..."
    
    # Create IAM role if it doesn't exist
    ROLE_NAME="synapty-lambda-role"
    ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text 2>/dev/null || echo "")
    
    if [ -z "$ROLE_ARN" ]; then
        echo "🔐 Creating IAM role..."
        
        # Create trust policy
        cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
        
        aws iam create-role \
            --role-name $ROLE_NAME \
            --assume-role-policy-document file://trust-policy.json
        
        # Attach basic execution policy
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        
        # Get role ARN
        ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)
        
        echo "⏳ Waiting for IAM role to propagate..."
        sleep 10
    fi
    
    # Create Lambda function
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime python3.9 \
        --role $ROLE_ARN \
        --handler simple_main.lambda_handler \
        --zip-file fileb://synapty-backend.zip \
        --timeout 30 \
        --memory-size 512 \
        --environment Variables="{OPENAI_API_KEY=$OPENAI_API_KEY}"
fi

# Create API Gateway
echo "🌐 Setting up API Gateway..."
API_NAME="synapty-api"

# Check if API exists
API_ID=$(aws apigatewayv2 get-apis --query "Items[?Name=='$API_NAME'].ApiId" --output text 2>/dev/null || echo "")

if [ -z "$API_ID" ] || [ "$API_ID" = "None" ]; then
    echo "🆕 Creating new API Gateway..."
    
    # Create API
    API_RESPONSE=$(aws apigatewayv2 create-api \
        --name $API_NAME \
        --protocol-type HTTP \
        --cors-configuration 'AllowCredentials=false,AllowHeaders=["content-type","x-amz-date","authorization","x-api-key","x-amz-security-token"],AllowMethods=["*"],AllowOrigins=["*"],MaxAge=86400')
    
    API_ID=$(echo $API_RESPONSE | grep -o '"ApiId":"[^"]*' | cut -d'"' -f4)
    API_ENDPOINT=$(echo $API_RESPONSE | grep -o '"ApiEndpoint":"[^"]*' | cut -d'"' -f4)
    
    # Create integration
    INTEGRATION_RESPONSE=$(aws apigatewayv2 create-integration \
        --api-id $API_ID \
        --integration-type AWS_PROXY \
        --integration-uri arn:aws:lambda:$AWS_REGION:$AWS_ACCOUNT_ID:function:$FUNCTION_NAME \
        --payload-format-version 2.0)
    
    INTEGRATION_ID=$(echo $INTEGRATION_RESPONSE | grep -o '"IntegrationId":"[^"]*' | cut -d'"' -f4)
    
    # Create routes
    aws apigatewayv2 create-route \
        --api-id $API_ID \
        --route-key 'ANY /{proxy+}' \
        --target integrations/$INTEGRATION_ID > /dev/null
    
    aws apigatewayv2 create-route \
        --api-id $API_ID \
        --route-key 'ANY /' \
        --target integrations/$INTEGRATION_ID > /dev/null
    
    # Create stage
    aws apigatewayv2 create-stage \
        --api-id $API_ID \
        --stage-name prod \
        --auto-deploy > /dev/null
    
    # Grant API Gateway permission to invoke Lambda
    aws lambda add-permission \
        --function-name $FUNCTION_NAME \
        --statement-id apigateway-access \
        --action lambda:InvokeFunction \
        --principal apigateway.amazonaws.com \
        --source-arn "arn:aws:execute-api:$AWS_REGION:$AWS_ACCOUNT_ID:$API_ID/*/*" > /dev/null
else
    echo "📝 Using existing API Gateway..."
    API_ENDPOINT=$(aws apigatewayv2 get-api --api-id $API_ID --query 'ApiEndpoint' --output text)
fi

FULL_API_URL="$API_ENDPOINT/prod"

echo "✅ Deployment completed successfully!"
echo "🔗 API Gateway URL: $FULL_API_URL"
echo ""
echo "🧪 Testing endpoints:"
echo "• Health check: curl $FULL_API_URL/api/health"
echo "• Generate graph: curl -X POST $FULL_API_URL/api/generate-graph -H 'Content-Type: application/json' -d '{\"topic\":\"test\"}'"
echo ""
echo "Next steps:"
echo "1. Update your frontend API_BASE_URL to: $FULL_API_URL"
echo "2. Test the deployment with the commands above"
echo "3. Deploy your frontend with the updated API URL"

# Cleanup
cd ..
rm -rf lambda-deploy

echo "🧹 Cleanup completed!"