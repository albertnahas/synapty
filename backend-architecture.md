# Synapti Backend Architecture

## Architecture Overview

### Technology Stack
- **Framework**: FastAPI 0.104+ with Pydantic v2 UV
- **Runtime**: Python 3.11 + AWS Lambda (Mangum ASGI adapter)
- **AI Service**: OpenAI GPT-4 with Structured Outputs
- **Caching**: Redis (AWS ElastiCache)
- **Storage**: DynamoDB + S3
- **Infrastructure**: AWS CDK for IaC

## 1. Service Architecture

### Lambda Function Structure
```
backend/
├── src/
│   ├── main.py                 # FastAPI app + Mangum handler
│   ├── api/                    # API route handlers
│   │   ├── __init__.py
│   │   ├── mindmap.py         # Mindmap generation endpoints
│   │   ├── export.py          # Export functionality
│   │   └── health.py          # Health checks
│   ├── services/              # Business logic
│   │   ├── __init__.py
│   │   ├── graph_generator.py # Graph generation service
│   │   ├── node_expander.py   # Node expansion service
│   │   ├── export_service.py  # Export service
│   │   └── openai_service.py  # OpenAI integration
│   ├── models/                # Data models
│   │   ├── __init__.py
│   │   ├── graph.py          # Graph data models
│   │   ├── requests.py       # API request models
│   │   └── responses.py      # API response models
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   ├── cache.py          # Redis caching
│   │   ├── database.py       # DynamoDB operations
│   │   └── validators.py     # Input validation
│   └── config/               # Configuration
│       ├── __init__.py
│       ├── settings.py       # Environment settings
│       └── prompts.py        # OpenAI prompts
├── requirements.txt          # Dependencies
└── serverless.yml           # Serverless config
```

### Core Application (`src/main.py`)
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from mangum import Mangum
import logging
import time
from contextlib import asynccontextmanager

from .api import mindmap, export, health
from .config.settings import Settings
from .utils.cache import CacheManager
from .utils.database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic for Lambda"""
    # Startup
    logger.info("FastAPI startup - initializing services")
    
    # Initialize connections
    cache_manager = CacheManager()
    db_manager = DatabaseManager()
    
    # Store in app state
    app.state.cache = cache_manager
    app.state.db = db_manager
    
    yield
    
    # Shutdown
    logger.info("FastAPI shutdown - cleaning up")

# Create FastAPI app
app = FastAPI(
    title="Synapti API",
    description="AI-Powered Dynamic Mindmap Learning Tool",
    version="0.1.0",
    docs_url="/docs" if Settings().environment != "production" else None,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings().cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware for request timing
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(mindmap.router, prefix="/v1/mindmap", tags=["mindmap"])
app.include_router(export.router, prefix="/v1/export", tags=["export"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Synapti API v0.1.0", "status": "healthy"}

# Lambda handler
handler = Mangum(app, lifespan="off")

def lambda_handler(event, context):
    """AWS Lambda handler with cold start optimization"""
    # Add custom headers for Lambda context
    if "headers" not in event:
        event["headers"] = {}
    
    event["headers"]["x-lambda-request-id"] = context.aws_request_id
    event["headers"]["x-lambda-remaining-time"] = str(context.get_remaining_time_in_millis())
    
    return handler(event, context)
```

### Graph Generation Service (`src/services/graph_generator.py`)
```python
from typing import List, Dict, Any, Optional
import asyncio
import json
import hashlib
from pydantic import BaseModel, Field

from ..models.graph import Graph, Node, Edge, GraphMetadata
from ..models.requests import GenerateGraphRequest
from ..services.openai_service import OpenAIService
from ..utils.cache import CacheManager
from ..utils.validators import validate_topic
from ..config.prompts import MINDMAP_GENERATION_PROMPT

class GraphGenerationService:
    def __init__(self, openai_service: OpenAIService, cache_manager: CacheManager):
        self.openai_service = openai_service
        self.cache_manager = cache_manager
        
    async def generate_graph(self, request: GenerateGraphRequest) -> Graph:
        """Generate mindmap graph from topic"""
        # Validate input
        validate_topic(request.topic)
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_graph = await self.cache_manager.get(cache_key)
        if cached_graph:
            return Graph.parse_obj(cached_graph)
        
        # Generate new graph
        start_time = time.time()
        
        try:
            # Call OpenAI for graph generation
            openai_response = await self.openai_service.generate_mindmap(
                topic=request.topic,
                max_nodes=request.max_nodes,
                style=request.style_preference,
                language=request.language
            )
            
            # Process response into graph structure
            graph = await self._process_openai_response(
                openai_response, 
                request.topic,
                start_time
            )
            
            # Cache the result
            await self.cache_manager.set(
                cache_key, 
                graph.dict(), 
                ttl=3600  # 1 hour
            )
            
            return graph
            
        except Exception as e:
            logger.error(f"Graph generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate mindmap: {str(e)}"
            )
    
    async def _process_openai_response(
        self, 
        openai_response: Dict[str, Any], 
        topic: str,
        start_time: float
    ) -> Graph:
        """Process OpenAI response into Graph model"""
        
        nodes = []
        edges = []
        
        # Create root node
        root_node = Node(
            id="root",
            title=topic,
            summary=f"Central concept: {topic}",
            position={"x": 0, "y": 0, "z": 0},
            color="#8B5CF6",
            size=1.5,
            expandable=False,
            connections=[]
        )
        nodes.append(root_node)
        
        # Process concept nodes from OpenAI
        concept_nodes = openai_response.get("nodes", [])
        node_positions = self._calculate_node_positions(len(concept_nodes))
        
        for i, concept in enumerate(concept_nodes):
            node_id = f"node_{i:03d}"
            
            node = Node(
                id=node_id,
                title=concept["title"],
                summary=concept["summary"],
                position=node_positions[i],
                color=self._get_node_color(concept.get("category", "general")),
                size=1.0 + (concept.get("importance", 0.5) * 0.5),
                expandable=True,
                connections=[]
            )
            nodes.append(node)
            
            # Create edge from root to node
            edge = Edge(
                id=f"edge_root_{node_id}",
                source="root",
                target=node_id,
                relationship="contains",
                strength=concept.get("importance", 0.7)
            )
            edges.append(edge)
            
            # Update connections
            root_node.connections.append(node_id)
        
        # Create metadata
        metadata = GraphMetadata(
            generation_time=time.time() - start_time,
            complexity_score=len(nodes) / 10.0,  # Simple complexity metric
            total_expandable_nodes=len([n for n in nodes if n.expandable]),
            ai_model_version="gpt-4-1106-preview"
        )
        
        return Graph(
            id=str(uuid.uuid4()),
            topic=topic,
            nodes=nodes,
            edges=edges,
            metadata=metadata,
            created_at=datetime.utcnow(),
            last_modified=datetime.utcnow()
        )
    
    def _calculate_node_positions(self, num_nodes: int) -> List[Dict[str, float]]:
        """Calculate 3D positions for nodes in circular layout"""
        positions = []
        radius = 4.0
        
        for i in range(num_nodes):
            angle = (2 * math.pi * i) / num_nodes
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            y = random.uniform(-1, 1)  # Add some vertical variation
            
            positions.append({"x": x, "y": y, "z": z})
        
        return positions
    
    def _get_node_color(self, category: str) -> str:
        """Get color based on node category"""
        color_map = {
            "concept": "#3B82F6",    # Blue
            "application": "#10B981", # Green
            "theory": "#8B5CF6",     # Purple
            "method": "#F59E0B",     # Orange
            "tool": "#EF4444",       # Red
            "general": "#6B7280"     # Gray
        }
        return color_map.get(category, "#6B7280")
    
    def _generate_cache_key(self, request: GenerateGraphRequest) -> str:
        """Generate cache key from request parameters"""
        key_data = {
            "topic": request.topic.lower().strip(),
            "max_nodes": request.max_nodes,
            "style": request.style_preference,
            "language": request.language
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"graph:{hashlib.md5(key_string.encode()).hexdigest()}"
```

### Node Expansion Service (`src/services/node_expander.py`)
```python
from typing import List, Dict, Any, Tuple
import asyncio
import time
from ..models.graph import Node, Edge
from ..models.requests import ExpandNodeRequest
from ..services.openai_service import OpenAIService
from ..utils.cache import CacheManager
from ..utils.database import DatabaseManager

class NodeExpansionService:
    def __init__(
        self, 
        openai_service: OpenAIService, 
        cache_manager: CacheManager,
        db_manager: DatabaseManager
    ):
        self.openai_service = openai_service
        self.cache_manager = cache_manager
        self.db_manager = db_manager
        
    async def expand_node(
        self, 
        graph_id: str, 
        node_id: str, 
        request: ExpandNodeRequest
    ) -> Tuple[List[Node], List[Edge]]:
        """Expand a node with sub-concepts"""
        
        # Get current graph state
        graph = await self.db_manager.get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail="Graph not found")
        
        # Find target node
        target_node = next((n for n in graph.nodes if n.id == node_id), None)
        if not target_node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        if not target_node.expandable:
            raise HTTPException(status_code=400, detail="Node is not expandable")
        
        # Check cache for expansion
        cache_key = f"expansion:{graph_id}:{node_id}"
        cached_expansion = await self.cache_manager.get(cache_key)
        if cached_expansion:
            return self._parse_cached_expansion(cached_expansion)
        
        try:
            # Generate expansion using OpenAI
            expansion_response = await self.openai_service.expand_node(
                node_title=target_node.title,
                node_summary=target_node.summary,
                context=self._build_context(graph, target_node),
                max_children=request.max_children,
                focus_areas=request.focus_areas
            )
            
            # Process expansion into nodes and edges
            new_nodes, new_edges = await self._process_expansion(
                expansion_response,
                target_node,
                request.expansion_depth
            )
            
            # Cache the expansion
            await self.cache_manager.set(
                cache_key,
                {
                    "nodes": [n.dict() for n in new_nodes],
                    "edges": [e.dict() for e in new_edges]
                },
                ttl=1800  # 30 minutes
            )
            
            # Update graph in database
            await self.db_manager.add_nodes_to_graph(graph_id, new_nodes, new_edges)
            
            return new_nodes, new_edges
            
        except Exception as e:
            logger.error(f"Node expansion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to expand node: {str(e)}"
            )
    
    async def _process_expansion(
        self,
        expansion_response: Dict[str, Any],
        parent_node: Node,
        depth: int
    ) -> Tuple[List[Node], List[Edge]]:
        """Process OpenAI expansion response into nodes and edges"""
        
        new_nodes = []
        new_edges = []
        
        sub_concepts = expansion_response.get("sub_concepts", [])
        child_positions = self._calculate_child_positions(
            parent_node.position,
            len(sub_concepts)
        )
        
        for i, concept in enumerate(sub_concepts):
            child_id = f"{parent_node.id}_{i:03d}"
            
            child_node = Node(
                id=child_id,
                title=concept["title"],
                summary=concept["summary"],
                position=child_positions[i],
                color=self._derive_child_color(parent_node.color),
                size=max(0.6, parent_node.size * 0.8),  # Smaller than parent
                expandable=concept.get("expandable", True) and depth > 1,
                parent_id=parent_node.id,
                connections=[]
            )
            new_nodes.append(child_node)
            
            # Create edge from parent to child
            edge = Edge(
                id=f"edge_{parent_node.id}_{child_id}",
                source=parent_node.id,
                target=child_id,
                relationship=concept.get("relationship", "contains"),
                strength=concept.get("strength", 0.8)
            )
            new_edges.append(edge)
        
        return new_nodes, new_edges
    
    def _calculate_child_positions(
        self, 
        parent_pos: Dict[str, float], 
        num_children: int
    ) -> List[Dict[str, float]]:
        """Calculate positions for child nodes around parent"""
        positions = []
        radius = 2.0
        
        for i in range(num_children):
            angle = (2 * math.pi * i) / num_children
            x = parent_pos["x"] + radius * math.cos(angle)
            z = parent_pos["z"] + radius * math.sin(angle)
            y = parent_pos["y"] + random.uniform(-0.5, 0.5)
            
            positions.append({"x": x, "y": y, "z": z})
        
        return positions
    
    def _build_context(self, graph: Graph, target_node: Node) -> str:
        """Build context string for OpenAI expansion"""
        context_parts = [
            f"Main topic: {graph.topic}",
            f"Expanding node: {target_node.title}",
            f"Node summary: {target_node.summary}"
        ]
        
        # Add related nodes context
        related_nodes = [
            n for n in graph.nodes 
            if n.id in target_node.connections or target_node.id in n.connections
        ]
        
        if related_nodes:
            related_titles = [n.title for n in related_nodes[:3]]  # Limit context
            context_parts.append(f"Related concepts: {', '.join(related_titles)}")
        
        return "\n".join(context_parts)
    
    def _derive_child_color(self, parent_color: str) -> str:
        """Generate child node color based on parent"""
        # Simple color derivation - could be more sophisticated
        color_variations = {
            "#3B82F6": "#60A5FA",  # Blue -> Lighter blue
            "#10B981": "#34D399",  # Green -> Lighter green
            "#8B5CF6": "#A78BFA",  # Purple -> Lighter purple
            "#F59E0B": "#FBBF24",  # Orange -> Lighter orange
            "#EF4444": "#F87171",  # Red -> Lighter red
        }
        return color_variations.get(parent_color, "#9CA3AF")
```

### OpenAI Service Integration (`src/services/openai_service.py`)
```python
from typing import Dict, Any, List, Optional
import asyncio
import json
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..config.settings import Settings
from ..config.prompts import (
    MINDMAP_GENERATION_PROMPT,
    NODE_EXPANSION_PROMPT,
    MINDMAP_SCHEMA,
    EXPANSION_SCHEMA
)

class OpenAIService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=Settings().openai_api_key)
        self.default_model = "gpt-4-1106-preview"
        
    async def generate_mindmap(
        self,
        topic: str,
        max_nodes: int = 8,
        style: str = "academic",
        language: str = "en"
    ) -> Dict[str, Any]:
        """Generate mindmap structure using OpenAI structured output"""
        
        prompt = MINDMAP_GENERATION_PROMPT.format(
            topic=topic,
            max_nodes=max_nodes,
            style_preference=style,
            language=language
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert educational content creator specialized in creating clear, structured mindmaps for learning."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "mindmap_generation",
                        "schema": MINDMAP_SCHEMA
                    }
                },
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"OpenAI mindmap generation failed: {e}")
            raise Exception(f"AI service error: {str(e)}")
    
    async def expand_node(
        self,
        node_title: str,
        node_summary: str,
        context: str,
        max_children: int = 5,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Expand a node with sub-concepts"""
        
        focus_areas_str = ", ".join(focus_areas) if focus_areas else "general concepts"
        
        prompt = NODE_EXPANSION_PROMPT.format(
            node_title=node_title,
            node_summary=node_summary,
            context=context,
            max_children=max_children,
            focus_areas=focus_areas_str
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert educational content creator. Expand concepts with clear, logical sub-concepts that maintain educational coherence."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "node_expansion",
                        "schema": EXPANSION_SCHEMA
                    }
                },
                temperature=0.6,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"OpenAI node expansion failed: {e}")
            raise Exception(f"AI expansion error: {str(e)}")
    
    async def validate_graph_coherence(
        self,
        graph_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that generated graph maintains educational coherence"""
        
        validation_prompt = f"""
        Analyze this mindmap structure for educational coherence:
        
        Topic: {graph_data.get('topic', 'Unknown')}
        Nodes: {len(graph_data.get('nodes', []))}
        
        Check for:
        1. Logical relationships between concepts
        2. Appropriate depth and breadth
        3. Educational value and clarity
        4. Missing critical concepts
        
        Return a coherence score (0-1) and suggestions for improvement.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an educational content validator. Assess mindmap structures for learning effectiveness."
                    },
                    {
                        "role": "user",
                        "content": validation_prompt
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "coherence_validation",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "coherence_score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1
                                },
                                "strengths": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "improvements": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "missing_concepts": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["coherence_score"]
                        }
                    }
                },
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"OpenAI validation failed: {e}")
            return {"coherence_score": 0.5, "error": str(e)}
```

## 2. AWS Lambda Optimization

### Cold Start Mitigation (`src/utils/lambda_optimizer.py`)
```python
import time
import asyncio
from typing import Dict, Any
import boto3
from concurrent.futures import ThreadPoolExecutor

class LambdaOptimizer:
    """Optimize Lambda performance and cold starts"""
    
    def __init__(self):
        self.warm_start_threshold = 100  # ms
        self.connection_pool = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        
    async def warm_up_connections(self) -> None:
        """Pre-warm external service connections"""
        tasks = [
            self._warm_openai_connection(),
            self._warm_redis_connection(),
            self._warm_dynamodb_connection()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _warm_openai_connection(self) -> None:
        """Pre-warm OpenAI client connection"""
        try:
            from ..services.openai_service import OpenAIService
            service = OpenAIService()
            # Simple health check call
            await service.client.models.list()
            logger.info("OpenAI connection warmed")
        except Exception as e:
            logger.warning(f"OpenAI warmup failed: {e}")
    
    async def _warm_redis_connection(self) -> None:
        """Pre-warm Redis connection"""
        try:
            from ..utils.cache import CacheManager
            cache = CacheManager()
            await cache.ping()
            logger.info("Redis connection warmed")
        except Exception as e:
            logger.warning(f"Redis warmup failed: {e}")
    
    async def _warm_dynamodb_connection(self) -> None:
        """Pre-warm DynamoDB connection"""
        try:
            from ..utils.database import DatabaseManager
            db = DatabaseManager()
            await db.health_check()
            logger.info("DynamoDB connection warmed")
        except Exception as e:
            logger.warning(f"DynamoDB warmup failed: {e}")
    
    def optimize_payload_size(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize response payload size"""
        # Remove unnecessary fields
        optimized = data.copy()
        
        # Compress position data
        if "nodes" in optimized:
            for node in optimized["nodes"]:
                if "position" in node:
                    pos = node["position"]
                    # Round to 2 decimal places
                    node["position"] = {
                        k: round(v, 2) for k, v in pos.items()
                    }
        
        return optimized
    
    def measure_cold_start(self, start_time: float) -> bool:
        """Determine if this was a cold start"""
        init_time = (time.time() - start_time) * 1000
        return init_time > self.warm_start_threshold

# Global instance
optimizer = LambdaOptimizer()
```

### Performance Monitoring (`src/utils/monitoring.py`)
```python
import time
import json
import boto3
from typing import Dict, Any, Optional
from functools import wraps
import logging

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.metrics_buffer = []
        
    def track_api_call(self, func):
        """Decorator to track API call performance"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            endpoint = func.__name__
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metrics
                duration = (time.time() - start_time) * 1000
                await self._record_metric(
                    'APICallDuration',
                    duration,
                    {'Endpoint': endpoint, 'Status': 'Success'}
                )
                
                return result
                
            except Exception as e:
                # Record error metrics
                duration = (time.time() - start_time) * 1000
                await self._record_metric(
                    'APICallDuration',
                    duration,
                    {'Endpoint': endpoint, 'Status': 'Error'}
                )
                
                await self._record_metric(
                    'APIErrors',
                    1,
                    {'Endpoint': endpoint, 'ErrorType': type(e).__name__}
                )
                
                raise
                
        return wrapper
    
    async def _record_metric(
        self,
        metric_name: str,
        value: float,
        dimensions: Dict[str, str]
    ) -> None:
        """Record CloudWatch metric"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace='Synapti/API',
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': 'Milliseconds' if 'Duration' in metric_name else 'Count',
                        'Dimensions': [
                            {'Name': k, 'Value': v} 
                            for k, v in dimensions.items()
                        ]
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")

# Global monitor instance
performance_monitor = PerformanceMonitor()
```

## 3. Caching Strategy

### Redis Cache Manager (`src/utils/cache.py`)
```python
import redis.asyncio as redis
import json
import pickle
import gzip
from typing import Any, Optional, Union
import logging
from ..config.settings import Settings

class CacheManager:
    """Redis-based caching with compression and serialization"""
    
    def __init__(self):
        self.redis_client = None
        self.compression_threshold = 1024  # Compress data > 1KB
        
    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client"""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                Settings().redis_url,
                encoding="utf-8",
                decode_responses=False,  # Handle binary data
                max_connections=10,
                retry_on_timeout=True
            )
        return self.redis_client
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        compress: bool = True
    ) -> bool:
        """Set cached value with optional compression"""
        try:
            client = await self._get_client()
            
            # Serialize data
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value).encode('utf-8')
            else:
                serialized = pickle.dumps(value)
            
            # Compress if beneficial
            if compress and len(serialized) > self.compression_threshold:
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized) * 0.9:  # 10% improvement
                    serialized = compressed
                    key = f"compressed:{key}"
            
            await client.setex(key, ttl, serialized)
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value with decompression"""
        try:
            client = await self._get_client()
            
            # Try compressed version first
            compressed_key = f"compressed:{key}"
            data = await client.get(compressed_key)
            is_compressed = True
            
            if data is None:
                data = await client.get(key)
                is_compressed = False
            
            if data is None:
                return None
            
            # Decompress if needed
            if is_compressed:
                data = gzip.decompress(data)
            
            # Deserialize
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete cached value"""
        try:
            client = await self._get_client()
            deleted = await client.delete(key, f"compressed:{key}")
            return deleted > 0
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            client = await self._get_client()
            return await client.exists(key, f"compressed:{key}") > 0
        except Exception as e:
            logger.error(f"Cache exists check failed for key {key}: {e}")
            return False
    
    async def ping(self) -> bool:
        """Health check for Redis connection"""
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            client = await self._get_client()
            info = await client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate"""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
```

## 4. Error Handling & Resilience

### Circuit Breaker Pattern (`src/utils/circuit_breaker.py`)
```python
import asyncio
import time
from enum import Enum
from typing import Callable, Any
import logging

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )

# Service-specific circuit breakers
openai_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=Exception
)

redis_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception
)
```

This backend architecture provides:
- **Scalability**: Serverless Lambda functions with auto-scaling
- **Performance**: <2s generation, <1s expansion with caching and optimization
- **Reliability**: Circuit breakers, error handling, health checks
- **Maintainability**: Clean service architecture with proper separation
- **Observability**: Comprehensive monitoring and logging
- **Cost Efficiency**: Pay-per-request serverless model