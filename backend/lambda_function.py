"""
AWS Lambda handler for Synapty backend
"""
from mangum import Mangum
from main import app

# Create the Lambda handler
handler = Mangum(app)

# Optional: Add custom error handling
def lambda_handler(event, context):
    """
    AWS Lambda handler with error handling
    """
    try:
        return handler(event, context)
    except Exception as e:
        print(f"Lambda error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            },
            "body": '{"error": "Internal server error"}'
        }