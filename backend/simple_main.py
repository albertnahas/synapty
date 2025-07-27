import json
import os
import requests
from typing import Dict, Any, List
import math

def lambda_handler(event, context):
    """
    Lambda handler for Synapty API using requests instead of openai library
    Also works for local development when called directly
    """
    try:
        # Parse the request - handle API Gateway v2, v1, Function URL, and local formats
        if 'requestContext' in event and 'http' in event['requestContext']:
            # API Gateway v2 format (HTTP API) or Function URL format
            method = event['requestContext']['http']['method']
            path = event['requestContext']['http']['path']
            if path.startswith('/prod'):
                # Remove /prod prefix from API Gateway v2
                path = path[5:] or '/'
        elif 'httpMethod' in event:
            # API Gateway v1 format (REST API)
            method = event['httpMethod']
            path = event['path']
        elif 'method' in event and 'path' in event:
            # Local development format
            method = event['method']
            path = event['path']
        else:
            # Direct invocation
            method = 'GET'
            path = '/'
        
        # CORS headers
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Content-Type": "application/json"
        }
        
        # Handle OPTIONS requests (CORS preflight)
        if method == 'OPTIONS':
            return {
                "statusCode": 200,
                "headers": headers,
                "body": ""
            }
        
        # Health check endpoint
        if path == '/api/health':
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({"status": "healthy", "version": "0.1.0"})
            }
        
        # Root endpoint
        if path == '/' or path == '':
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({"message": "Synapty API is running"})
            }
        
        # Generate graph endpoint
        if path == '/api/generate-graph' and method == 'POST':
            body = json.loads(event.get('body', '{}'))
            topic = body.get('topic', '')
            
            if not topic:
                return {
                    "statusCode": 400,
                    "headers": headers,
                    "body": json.dumps({"error": "Topic is required"})
                }
            
            # Generate graph using OpenAI API directly via requests
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return {
                        "statusCode": 500,
                        "headers": headers,
                        "body": json.dumps({"error": "OpenAI API key not configured"})
                    }
                
                prompt = f"""
                Create a mindmap for the topic: "{topic}"
                
                Generate 6-8 key concepts that are the most important aspects of this topic.
                For each concept, provide:
                1. A concise title (2-4 words)
                2. A brief summary (1-2 sentences)
                
                Return the response as a JSON object with this structure:
                {{
                    "concepts": [
                        {{
                            "title": "Concept Title",
                            "summary": "Brief explanation of this concept"
                        }}
                    ]
                }}
                """
                
                # Call OpenAI API directly
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4",
                        "messages": [
                            {"role": "system", "content": "You are an expert educator who creates clear, structured learning materials. Always respond with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1000
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    return {
                        "statusCode": 500,
                        "headers": headers,
                        "body": json.dumps({"error": f"OpenAI API error: {response.status_code}"})
                    }
                
                # Parse the response
                openai_response = response.json()
                content = openai_response['choices'][0]['message']['content']
                data = json.loads(content)
                
                # Convert to graph structure
                nodes = []
                edges = []
                
                # Create center node
                center_node = {
                    "id": "center",
                    "title": topic,
                    "summary": f"Main topic: {topic}",
                    "position": {"x": 0, "y": 0, "z": 0},
                    "children": [],
                    "level": 0
                }
                nodes.append(center_node)
                
                # Create concept nodes in a circle around center
                num_concepts = len(data["concepts"])
                radius = 300
                
                for i, concept in enumerate(data["concepts"]):
                    angle = (2 * math.pi * i) / num_concepts
                    x = radius * math.cos(angle)
                    z = radius * math.sin(angle)
                    
                    node_id = f"node_{i}"
                    node = {
                        "id": node_id,
                        "title": concept["title"],
                        "summary": concept["summary"],
                        "position": {"x": x, "y": 0, "z": z},
                        "children": [],
                        "parent": "center",
                        "level": 1
                    }
                    nodes.append(node)
                    
                    # Create edge from center to this node
                    edge = {
                        "id": f"edge_center_{node_id}",
                        "source": "center",
                        "target": node_id
                    }
                    edges.append(edge)
                
                graph_response = {
                    "nodes": nodes,
                    "edges": edges,
                    "center_topic": topic
                }
                
                return {
                    "statusCode": 200,
                    "headers": headers,
                    "body": json.dumps(graph_response)
                }
                
            except Exception as e:
                return {
                    "statusCode": 500,
                    "headers": headers,
                    "body": json.dumps({"error": f"Error generating graph: {str(e)}"})
                }
        
        # Expand node endpoint
        if path == '/api/expand-node' and method == 'POST':
            body = json.loads(event.get('body', '{}'))
            node_id = body.get('node_id', '')
            topic = body.get('topic', '')
            
            if not node_id or not topic:
                return {
                    "statusCode": 400,
                    "headers": headers,
                    "body": json.dumps({"error": "node_id and topic are required"})
                }
            
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return {
                        "statusCode": 500,
                        "headers": headers,
                        "body": json.dumps({"error": "OpenAI API key not configured"})
                    }
                
                prompt = f"""
                For the concept "{topic}"
                
                Generate 3-5 sub-concepts that dive deeper into this topic.
                For each sub-concept, provide:
                1. A concise title (2-4 words)
                2. A brief summary (1-2 sentences)
                
                Return the response as a JSON object with this structure:
                {{
                    "subconcepts": [
                        {{
                            "title": "Subconcept Title",
                            "summary": "Brief explanation of this subconcept"
                        }}
                    ]
                }}
                """
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4",
                        "messages": [
                            {"role": "system", "content": "You are an expert educator who creates clear, structured learning materials. Always respond with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 800
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    return {
                        "statusCode": 500,
                        "headers": headers,
                        "body": json.dumps({"error": f"OpenAI API error: {response.status_code}"})
                    }
                
                openai_response = response.json()
                content = openai_response['choices'][0]['message']['content']
                data = json.loads(content)
                
                # Convert expansion data to nodes and edges
                nodes = []
                edges = []
                
                num_subconcepts = len(data["subconcepts"])
                radius = 150
                
                for i, subconcept in enumerate(data["subconcepts"]):
                    angle = (2 * math.pi * i) / num_subconcepts
                    x = radius * math.cos(angle)
                    z = radius * math.sin(angle)
                    
                    child_node_id = f"{node_id}_child_{i}"
                    node = {
                        "id": child_node_id,
                        "title": subconcept["title"],
                        "summary": subconcept["summary"],
                        "position": {"x": x, "y": 0, "z": z},
                        "children": [],
                        "parent": node_id,
                        "level": 2
                    }
                    nodes.append(node)
                    
                    # Create edge from parent to child
                    edge = {
                        "id": f"edge_{node_id}_{child_node_id}",
                        "source": node_id,
                        "target": child_node_id
                    }
                    edges.append(edge)
                
                expansion_response = {
                    "nodes": nodes,
                    "edges": edges,
                    "expanded_node": node_id
                }
                
                return {
                    "statusCode": 200,
                    "headers": headers,
                    "body": json.dumps(expansion_response)
                }
                
            except Exception as e:
                return {
                    "statusCode": 500,
                    "headers": headers,
                    "body": json.dumps({"error": f"Error expanding node: {str(e)}"})
                }
        
        # 404 for unknown endpoints
        return {
            "statusCode": 404,
            "headers": headers,
            "body": json.dumps({"error": "Not found"})
        }
        
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "application/json"
            },
            "body": json.dumps({"error": f"Internal server error: {str(e)}"})
        }


# Local development server
if __name__ == "__main__":
    from dotenv import load_dotenv
    import uvicorn
    from fastapi import FastAPI, Request
    
    # Load environment variables for local development
    load_dotenv()
    
    app = FastAPI(title="Synapty API", version="0.1.0")
    
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"])
    async def catch_all(request: Request, path: str):
        """Route all requests to the lambda handler"""
        
        # Convert FastAPI request to Lambda event format
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            body = body.decode() if body else "{}"
        
        event = {
            "method": request.method,
            "path": f"/{path}" if path else "/",
            "body": body,
            "headers": dict(request.headers)
        }
        
        # Call the lambda handler
        response = lambda_handler(event, {})
        
        # Convert Lambda response to FastAPI response
        from fastapi import Response
        return Response(
            content=response["body"],
            status_code=response["statusCode"],
            headers=response["headers"],
            media_type="application/json"
        )
    
    print("🚀 Starting Synapty API server on http://localhost:8000")
    print("🔍 Health check: http://localhost:8000/api/health")
    print("📝 Test endpoint: curl -X POST http://localhost:8000/api/generate-graph -H 'Content-Type: application/json' -d '{\"topic\":\"test\"}'")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)