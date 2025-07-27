# Synapty Deployment Guide

This document contains all the necessary information to run and deploy Synapty when making changes.

## Overview

Synapty is an AI-powered dynamic mindmap learning tool with:
- **Backend**: Python Lambda function using OpenAI GPT-4 API
- **Frontend**: React with Three.js 3D visualization
- **Deployment**: AWS Lambda + API Gateway

## Prerequisites

- AWS CLI configured with credentials for account `653858369289`
- Node.js and npm for frontend development
- OpenAI API key

## Current Deployment Configuration

### Backend (AWS Lambda)
- **Function Name**: `synapty-backend`
- **Runtime**: Python 3.9
- **Handler**: `simple_main.lambda_handler`
- **API Gateway**: HTTP API with CORS enabled
- **Current URL**: `https://sb6y93gdh1.execute-api.us-east-1.amazonaws.com/prod`

### Frontend
- **Framework**: React with TypeScript
- **3D Engine**: Three.js with React Three Fiber
- **API Configuration**: Points to AWS API Gateway URL

## Quick Deployment

### Automated Backend Deployment

Use the deployment script for complete backend setup:

```bash
cd /Users/albert.nahas/Synapty
./scripts/deploy-lambda.sh synapty-backend YOUR_OPENAI_API_KEY
```

**What the script does:**
- Creates a simplified Lambda function using only the `requests` library
- Installs dependencies and packages the function
- Creates/updates Lambda function with environment variables
- Sets up API Gateway HTTP API with CORS
- Configures routes and permissions
- Provides testing endpoints

### Frontend Deployment

Build and deploy the frontend:

```bash
cd frontend
npm install
npm run build
```

Deploy the `dist/` folder to your preferred hosting service.

## API Endpoints

### Health Check
```bash
GET /api/health
```
Returns: `{"status": "healthy", "version": "0.1.0"}`

### Generate Graph
```bash
POST /api/generate-graph
Content-Type: application/json

{
  "topic": "Machine Learning"
}
```

### Expand Node
```bash
POST /api/expand-node
Content-Type: application/json

{
  "node_id": "node_1",
  "topic": "Neural Networks"
}
```

## Testing Deployment

After deployment, test the endpoints:

```bash
# Health check
curl https://sb6y93gdh1.execute-api.us-east-1.amazonaws.com/prod/api/health

# Generate graph
curl -X POST https://sb6y93gdh1.execute-api.us-east-1.amazonaws.com/prod/api/generate-graph \
  -H "Content-Type: application/json" \
  -d '{"topic":"test"}'
```

## Making Changes

### Backend Changes
1. **Local Development**: Modify `backend/main.py` and test locally:
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
   ```
2. **Update Lambda Code**: Copy changes to `backend/simple_main.py` 
3. **Deploy to Lambda**: Run the deployment script:
   ```bash
   ./scripts/deploy-lambda.sh synapty-backend YOUR_API_KEY
   ```
3. Test the updated endpoints

### Frontend Changes
1. Make changes to React components in `frontend/src/`
2. Update API configuration in `frontend/.env` if the API Gateway URL changes
3. Build and deploy: `npm run build`

### Updating API URL
If the API Gateway URL changes after redeployment:
1. Update `frontend/.env`: `VITE_API_BASE_URL=new-api-url`
2. Rebuild frontend: `npm run build`
3. Redeploy frontend

## Important Files

### Backend Code
- **Local Development**: `/Users/albert.nahas/Synapty/backend/main.py`
  - Run locally: `python main.py` (requires FastAPI for dev server)
- **Lambda Deployment**: `/Users/albert.nahas/Synapty/backend/simple_main.py`  
  - Used by deployment script for AWS Lambda
  - Handler: `simple_main.lambda_handler`
- **Dependencies**: `requirements.txt` (requests + optional dev dependencies)

### Deployment Script
- **Location**: `/Users/albert.nahas/Synapty/scripts/deploy-lambda.sh`
- **Purpose**: Complete AWS Lambda deployment automation
- **Usage**: `./deploy-lambda.sh [function-name] [openai-api-key]`

### Frontend API Configuration
- **Location**: `/Users/albert.nahas/Synapty/frontend/.env`
- **Environment Variable**: `VITE_API_BASE_URL`
- **Current API URL**: `https://sb6y93gdh1.execute-api.us-east-1.amazonaws.com/prod`
- **Update the .env file** if the API Gateway URL changes

### Environment Variables
- **Backend**: `OPENAI_API_KEY` set in Lambda environment
- **Frontend**: `VITE_API_BASE_URL` in `.env` file (defaults to localhost for development)

## Troubleshooting

### Common Issues

1. **"Forbidden" Error**: Usually indicates API Gateway configuration issues
   - Check Lambda permissions for API Gateway
   - Verify CORS configuration

2. **OpenAI API Errors**: 
   - Verify API key is correctly set in Lambda environment
   - Check API key has sufficient credits

3. **CORS Issues**:
   - Ensure API Gateway has CORS enabled
   - Check headers in Lambda response

### Logs and Debugging

- **Lambda Logs**: Check CloudWatch logs for the `synapty-backend` function
- **Network Issues**: Use browser dev tools to inspect API calls
- **API Gateway**: Check API Gateway logs in CloudWatch

### Debug Commands

```bash
# Check Lambda logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/synapty

# Test Lambda function
aws lambda invoke --function-name synapty-backend --payload '{}' response.json
```

## Production Considerations

- **API Keys**: Store OpenAI API key securely (current: environment variable)
- **CORS**: Currently allows all origins (`*`) - restrict for production
- **Rate Limiting**: Consider implementing rate limiting for API endpoints
- **Monitoring**: Set up CloudWatch alarms for Lambda errors and API Gateway metrics

## Current Status

✅ **Working Configuration**:
- Backend deployed to AWS Lambda with API Gateway
- Frontend configured to use production API
- All endpoints functional with OpenAI integration
- CORS properly configured for cross-origin requests