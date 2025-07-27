#!/bin/bash

# Synapty Frontend Deployment Script
# Usage: ./deploy-frontend.sh [api-url] [deployment-target]

set -e

API_URL=${1}
DEPLOYMENT_TARGET=${2:-vercel}

if [ -z "$API_URL" ]; then
    echo "Usage: ./deploy-frontend.sh [api-url] [deployment-target]"
    echo "Example: ./deploy-frontend.sh https://abc123.lambda-url.us-east-1.on.aws/ vercel"
    echo "Deployment targets: vercel, s3, netlify"
    exit 1
fi

echo "🚀 Deploying Synapty frontend..."

# Update API URL in frontend
echo "🔧 Updating API configuration..."
cd frontend

# Create backup of original api.ts
cp src/api.ts src/api.ts.backup

# Update API_BASE_URL
sed -i.bak "s|const API_BASE_URL = '.*';|const API_BASE_URL = '$API_URL';|g" src/api.ts
rm src/api.ts.bak

echo "✅ Updated API URL to: $API_URL"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📥 Installing dependencies..."
    npm install
fi

# Build the project
echo "🏗️  Building frontend..."
npm run build

case $DEPLOYMENT_TARGET in
    vercel)
        echo "📤 Deploying to Vercel..."
        
        # Check if Vercel CLI is installed
        if ! command -v vercel &> /dev/null; then
            echo "Installing Vercel CLI..."
            npm install -g vercel
        fi
        
        # Deploy to Vercel
        vercel --prod --yes
        
        echo "✅ Deployed to Vercel!"
        ;;
        
    s3)
        echo "📤 Deploying to AWS S3..."
        
        BUCKET_NAME="synapty-frontend-$(date +%s)"
        
        # Create S3 bucket
        aws s3 mb s3://$BUCKET_NAME
        
        # Configure for static website hosting
        aws s3 website s3://$BUCKET_NAME \
            --index-document index.html \
            --error-document index.html
        
        # Create bucket policy for public read
        cat > bucket-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::$BUCKET_NAME/*"
    }
  ]
}
EOF
        
        aws s3api put-bucket-policy --bucket $BUCKET_NAME --policy file://bucket-policy.json
        
        # Upload files
        aws s3 sync dist/ s3://$BUCKET_NAME --delete
        
        # Get website URL
        WEBSITE_URL="http://$BUCKET_NAME.s3-website-$(aws configure get region).amazonaws.com"
        
        echo "✅ Deployed to S3!"
        echo "🔗 Website URL: $WEBSITE_URL"
        
        rm bucket-policy.json
        ;;
        
    netlify)
        echo "📤 Deploying to Netlify..."
        
        # Check if Netlify CLI is installed
        if ! command -v netlify &> /dev/null; then
            echo "Installing Netlify CLI..."
            npm install -g netlify-cli
        fi
        
        # Deploy to Netlify
        netlify deploy --prod --dir=dist
        
        echo "✅ Deployed to Netlify!"
        ;;
        
    *)
        echo "❌ Unknown deployment target: $DEPLOYMENT_TARGET"
        echo "Supported targets: vercel, s3, netlify"
        exit 1
        ;;
esac

# Restore original api.ts
mv src/api.ts.backup src/api.ts

echo "🧹 Restored original API configuration"
echo "✅ Frontend deployment completed!"

cd ..