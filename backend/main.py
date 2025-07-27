from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import openai
import os
from dotenv import load_dotenv
import json
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI(title="Synapty API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pydantic models
class Node(BaseModel):
    id: str
    title: str
    summary: str
    position: Dict[str, float]  # x, y, z coordinates
    children: List[str] = []
    parent: Optional[str] = None
    level: int = 0

class Edge(BaseModel):
    id: str
    source: str
    target: str

class GraphRequest(BaseModel):
    topic: str

class GraphResponse(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    center_topic: str

class ExpandRequest(BaseModel):
    node_id: str
    topic: str

# Graph generation agent
class GraphGenerationAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def generate_initial_graph(self, topic: str) -> GraphResponse:
        """Generate initial graph with 6-8 root nodes"""
        
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
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert educator who creates clear, structured learning materials. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the response
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Convert to graph structure
            nodes = []
            edges = []
            
            # Create center node
            center_node = Node(
                id="center",
                title=topic,
                summary=f"Main topic: {topic}",
                position={"x": 0, "y": 0, "z": 0},
                level=0
            )
            nodes.append(center_node)
            
            # Create concept nodes in a circle around center
            import math
            num_concepts = len(data["concepts"])
            radius = 300
            
            for i, concept in enumerate(data["concepts"]):
                angle = (2 * math.pi * i) / num_concepts
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
                
                node_id = f"node_{i}"
                node = Node(
                    id=node_id,
                    title=concept["title"],
                    summary=concept["summary"],
                    position={"x": x, "y": 0, "z": z},
                    parent="center",
                    level=1
                )
                nodes.append(node)
                
                # Create edge from center to this node
                edge = Edge(
                    id=f"edge_center_{node_id}",
                    source="center",
                    target=node_id
                )
                edges.append(edge)
            
            return GraphResponse(
                nodes=nodes,
                edges=edges,
                center_topic=topic
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating graph: {str(e)}")
    
    async def expand_node(self, node_id: str, topic: str, parent_summary: str) -> Dict[str, Any]:
        """Expand a node with 3-5 sub-concepts"""
        
        prompt = f"""
        For the concept "{topic}" with context: "{parent_summary}"
        
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
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert educator who creates clear, structured learning materials. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            return data
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error expanding node: {str(e)}")

# Initialize agent
graph_agent = GraphGenerationAgent()

@app.get("/")
async def root():
    return {"message": "Synapty API is running"}

@app.post("/api/generate-graph", response_model=GraphResponse)
async def generate_graph(request: GraphRequest):
    """Generate initial mindmap graph for a topic"""
    return await graph_agent.generate_initial_graph(request.topic)

@app.post("/api/expand-node")
async def expand_node(request: ExpandRequest):
    """Expand a node with sub-concepts"""
    # This would typically fetch the parent node from storage
    # For MVP, we'll pass the topic directly
    expansion_data = await graph_agent.expand_node(
        request.node_id, 
        request.topic,
        f"Expanding topic: {request.topic}"
    )
    
    # Convert expansion data to nodes and edges
    nodes = []
    edges = []
    
    import math
    num_subconcepts = len(expansion_data["subconcepts"])
    radius = 150
    
    for i, subconcept in enumerate(expansion_data["subconcepts"]):
        angle = (2 * math.pi * i) / num_subconcepts
        # Position relative to parent (would need parent position in real implementation)
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        
        child_node_id = f"{request.node_id}_child_{i}"
        node = Node(
            id=child_node_id,
            title=subconcept["title"],
            summary=subconcept["summary"],
            position={"x": x, "y": 0, "z": z},
            parent=request.node_id,
            level=2  # Assuming parent is level 1
        )
        nodes.append(node)
        
        # Create edge from parent to child
        edge = Edge(
            id=f"edge_{request.node_id}_{child_node_id}",
            source=request.node_id,
            target=child_node_id
        )
        edges.append(edge)
    
    return {
        "nodes": nodes,
        "edges": edges,
        "expanded_node": request.node_id
    }

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)