# Synapti Data Models & Database Design

## 1. Core Data Models

### Graph Model (`src/models/graph.py`)
```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid

class Position3D(BaseModel):
    """3D position coordinates"""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate") 
    z: float = Field(..., description="Z coordinate")

class NodeMetadata(BaseModel):
    """Additional node metadata"""
    category: str = Field(default="general", description="Node category")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")
    complexity: float = Field(default=0.5, ge=0.0, le=1.0, description="Complexity score")
    learning_level: str = Field(default="intermediate", description="Target learning level")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    
class Node(BaseModel):
    """Individual mindmap node"""
    id: str = Field(..., description="Unique node identifier")
    title: str = Field(..., min_length=1, max_length=100, description="Node title")
    summary: str = Field(..., min_length=1, max_length=500, description="Node summary")
    position: Position3D = Field(..., description="3D position")
    color: str = Field(..., regex=r'^#[0-9A-Fa-f]{6}$', description="Hex color code")
    size: float = Field(default=1.0, ge=0.1, le=3.0, description="Node size")
    expandable: bool = Field(default=True, description="Can be expanded")
    parent_id: Optional[str] = Field(None, description="Parent node ID")
    connections: List[str] = Field(default_factory=list, description="Connected node IDs")
    metadata: Optional[NodeMetadata] = Field(None, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('id')
    def validate_id(cls, v):
        """Validate node ID format"""
        if not v or len(v) < 3:
            raise ValueError('Node ID must be at least 3 characters')
        return v

class EdgeStyle(BaseModel):
    """Edge visual styling"""
    color: str = Field(default="#64748B", regex=r'^#[0-9A-Fa-f]{6}$')
    thickness: float = Field(default=1.0, ge=0.1, le=5.0)
    style: str = Field(default="solid", regex="^(solid|dashed|dotted)$")
    animated: bool = Field(default=False)

class Edge(BaseModel):
    """Connection between nodes"""
    id: str = Field(..., description="Unique edge identifier")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relationship: str = Field(
        default="contains",
        regex="^(contains|implements|relates_to|depends_on|leads_to|part_of)$",
        description="Relationship type"
    )
    strength: float = Field(default=0.8, ge=0.0, le=1.0, description="Connection strength")
    style: Optional[EdgeStyle] = Field(None, description="Visual styling")
    bidirectional: bool = Field(default=False, description="Two-way relationship")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class GraphMetadata(BaseModel):
    """Graph generation metadata"""
    generation_time: float = Field(..., ge=0.0, description="Generation time in seconds")
    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Graph complexity")
    total_expandable_nodes: int = Field(..., ge=0, description="Number of expandable nodes")
    ai_model_version: str = Field(..., description="AI model used")
    prompt_version: str = Field(default="1.0", description="Prompt template version")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality assessment")

class UserInteractions(BaseModel):
    """User interaction tracking"""
    expansions_count: int = Field(default=0, ge=0, description="Number of expansions")
    time_spent: int = Field(default=0, ge=0, description="Time spent in seconds")
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    export_count: int = Field(default=0, ge=0, description="Number of exports")
    nodes_created: int = Field(default=0, ge=0, description="Nodes created by user")

class Graph(BaseModel):
    """Complete mindmap graph"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique graph ID")
    topic: str = Field(..., min_length=2, max_length=200, description="Main topic")
    nodes: List[Node] = Field(..., min_items=1, description="Graph nodes")
    edges: List[Edge] = Field(default_factory=list, description="Graph edges")
    metadata: GraphMetadata = Field(..., description="Generation metadata")
    user_interactions: UserInteractions = Field(default_factory=UserInteractions)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1, ge=1, description="Graph version")
    
    @validator('nodes')
    def validate_nodes(cls, v):
        """Validate node relationships"""
        node_ids = {node.id for node in v}
        
        # Check for duplicate IDs
        if len(node_ids) != len(v):
            raise ValueError('Duplicate node IDs found')
        
        # Validate parent-child relationships
        for node in v:
            if node.parent_id and node.parent_id not in node_ids:
                raise ValueError(f'Parent node {node.parent_id} not found')
                
        return v
    
    @validator('edges')
    def validate_edges(cls, v, values):
        """Validate edge relationships"""
        if 'nodes' not in values:
            return v
            
        node_ids = {node.id for node in values['nodes']}
        
        for edge in v:
            if edge.source not in node_ids:
                raise ValueError(f'Source node {edge.source} not found')
            if edge.target not in node_ids:
                raise ValueError(f'Target node {edge.target} not found')
                
        return v
    
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        return next((node for node in self.nodes if node.id == node_id), None)
    
    def get_children(self, parent_id: str) -> List[Node]:
        """Get child nodes of a parent"""
        return [node for node in self.nodes if node.parent_id == parent_id]
    
    def get_connected_nodes(self, node_id: str) -> List[Node]:
        """Get nodes connected to given node"""
        connected_ids = set()
        
        # Direct connections from node
        node = self.get_node_by_id(node_id)
        if node:
            connected_ids.update(node.connections)
        
        # Connections via edges
        for edge in self.edges:
            if edge.source == node_id:
                connected_ids.add(edge.target)
            elif edge.target == node_id and edge.bidirectional:
                connected_ids.add(edge.source)
        
        return [self.get_node_by_id(id) for id in connected_ids if self.get_node_by_id(id)]
```

### Request/Response Models (`src/models/api.py`)
```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum

class StylePreference(str, Enum):
    """Style preferences for mindmap generation"""
    ACADEMIC = "academic"
    CASUAL = "casual"
    VISUAL = "visual"
    TECHNICAL = "technical"
    CREATIVE = "creative"

class ExportFormat(str, Enum):
    """Export format options"""
    JSON = "json"
    PPTX = "pptx"
    GOOGLE_SLIDES = "google_slides"
    SVG = "svg"
    PNG = "png"

class GenerateGraphRequest(BaseModel):
    """Request to generate new mindmap"""
    topic: str = Field(..., min_length=2, max_length=200, description="Topic to generate mindmap for")
    max_nodes: int = Field(default=8, ge=4, le=12, description="Maximum number of root nodes")
    depth_level: int = Field(default=1, ge=1, le=3, description="Initial depth level")
    style_preference: StylePreference = Field(default=StylePreference.ACADEMIC)
    language: str = Field(default="en", regex="^[a-z]{2}$", description="ISO language code")
    
    @validator('topic')
    def validate_topic(cls, v):
        """Validate topic content"""
        # Remove extra whitespace
        v = ' '.join(v.split())
        
        # Check for inappropriate content (basic validation)
        blocked_terms = ['spam', 'test123', 'asdf']
        if any(term in v.lower() for term in blocked_terms):
            raise ValueError('Invalid topic content')
            
        return v

class ExpandNodeRequest(BaseModel):
    """Request to expand a specific node"""
    node_id: str = Field(..., description="ID of node to expand")
    expansion_depth: int = Field(default=1, ge=1, le=2, description="Expansion depth")
    max_children: int = Field(default=5, ge=2, le=8, description="Maximum child nodes")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")

class ExportOptions(BaseModel):
    """Export configuration options"""
    include_metadata: bool = Field(default=True, description="Include generation metadata")
    layout_style: str = Field(
        default="hierarchical",
        regex="^(hierarchical|radial|force_directed)$",
        description="Layout style for export"
    )
    theme: str = Field(
        default="light",
        regex="^(light|dark|custom)$",
        description="Color theme"
    )
    slide_template: Optional[str] = Field(
        None,
        regex="^(minimal|detailed|presentation)$",
        description="Slide template for PPTX"
    )
    image_resolution: Optional[str] = Field(
        None,
        regex="^(low|medium|high)$",
        description="Image resolution for PNG/SVG"
    )

class ExportRequest(BaseModel):
    """Request to export mindmap"""
    graph_id: str = Field(..., description="Graph ID to export")
    format: ExportFormat = Field(..., description="Export format")
    options: ExportOptions = Field(default_factory=ExportOptions)

# Response Models
class GenerateGraphResponse(BaseModel):
    """Response for graph generation"""
    success: bool = Field(..., description="Operation success")
    data: Optional[Dict[str, Any]] = Field(None, description="Graph data")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details")

class ExpandNodeResponse(BaseModel):
    """Response for node expansion"""
    success: bool = Field(..., description="Operation success")
    data: Optional[Dict[str, Any]] = Field(None, description="Expansion data")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details")

class ExportResponse(BaseModel):
    """Response for export operation"""
    success: bool = Field(..., description="Operation success")
    data: Optional[Dict[str, Any]] = Field(None, description="Export data")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details")

class ErrorDetail(BaseModel):
    """Standardized error response"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human readable message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: str = Field(..., description="Request tracking ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

## 2. Database Schema Design

### DynamoDB Table Design

#### Primary Table: `synapti-graphs`
```yaml
Table: synapti-graphs
Partition Key: graph_id (String)
Sort Key: None (Single-item per graph)

Attributes:
  - graph_id: String (UUID)
  - topic: String
  - nodes: List (JSON serialized)
  - edges: List (JSON serialized) 
  - metadata: Map
  - user_interactions: Map
  - created_at: String (ISO format)
  - last_modified: String (ISO format)
  - version: Number
  - ttl: Number (Unix timestamp for auto-deletion)

Global Secondary Indexes:
  1. TopicIndex
     - Partition Key: topic_hash (String - MD5 of normalized topic)
     - Sort Key: created_at (String)
     - Projection: Keys + metadata
     
  2. TimeIndex  
     - Partition Key: date_partition (String - YYYY-MM-DD)
     - Sort Key: created_at (String)
     - Projection: Keys + topic + metadata

Local Secondary Index: None

Capacity:
  - Read: 5 RCU (On-demand auto-scaling)
  - Write: 5 WCU (On-demand auto-scaling)
  
TTL: 7 days for inactive graphs
```

#### Cache Table: `synapti-cache`
```yaml
Table: synapti-cache
Partition Key: cache_key (String)
Sort Key: None

Attributes:
  - cache_key: String
  - cache_value: String (JSON serialized)
  - cache_type: String (graph|expansion|export)
  - created_at: String
  - ttl: Number (Unix timestamp)

Capacity:
  - Read: 10 RCU (Burst scaling)
  - Write: 5 WCU (Burst scaling)
  
TTL: Based on cache_type (1h-24h)
```

#### Session Table: `synapti-sessions`
```yaml  
Table: synapti-sessions
Partition Key: session_id (String)
Sort Key: None

Attributes:
  - session_id: String (UUID)
  - graph_ids: List (Associated graphs)
  - user_agent: String
  - ip_address: String (Hashed for privacy)
  - created_at: String
  - last_activity: String
  - interaction_count: Number
  - ttl: Number

Capacity:
  - Read: 2 RCU
  - Write: 2 WCU
  
TTL: 24 hours
```

### Database Access Patterns

#### Pattern 1: Graph Generation & Retrieval
```python
# Create new graph
PUT synapti-graphs
{
  "graph_id": "uuid-4-string",
  "topic": "Machine Learning",
  "nodes": [...],
  "edges": [...],
  "metadata": {...},
  "created_at": "2025-07-27T10:30:00Z",
  "ttl": 1722159600
}

# Retrieve graph
GET synapti-graphs
Key: {"graph_id": "uuid-4-string"}

# Query by topic (for caching)
QUERY TopicIndex
{
  "topic_hash": "md5_hash_of_topic",
  "created_at": {">=": "2025-07-27T00:00:00Z"}
}
```

#### Pattern 2: Node Expansion Updates
```python
# Update graph with new nodes
UPDATE synapti-graphs
{
  "graph_id": "uuid-4-string",
  "ADD": {
    "nodes": [...new_nodes],
    "edges": [...new_edges]
  },
  "SET": {
    "last_modified": "2025-07-27T10:35:00Z",
    "version": "version + 1"
  }
}
```

#### Pattern 3: Caching Operations
```python
# Cache graph generation
PUT synapti-cache
{
  "cache_key": "graph:topic_hash",
  "cache_value": "{...json...}",
  "cache_type": "graph",
  "ttl": 1722159600
}

# Cache node expansion
PUT synapti-cache
{
  "cache_key": "expansion:graph_id:node_id",
  "cache_value": "{...expansion_data...}",
  "cache_type": "expansion", 
  "ttl": 1722157800
}
```

## 3. Database Operations (`src/utils/database.py`)
```python
import boto3
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import hashlib
from botocore.exceptions import ClientError

from ..models.graph import Graph, Node, Edge
from ..config.settings import Settings

class DatabaseManager:
    """DynamoDB operations manager"""
    
    def __init__(self):
        self.dynamodb = boto3.resource(
            'dynamodb',
            region_name=Settings().aws_region
        )
        self.graphs_table = self.dynamodb.Table('synapti-graphs')
        self.cache_table = self.dynamodb.Table('synapti-cache')
        self.sessions_table = self.dynamodb.Table('synapti-sessions')
        
    async def save_graph(self, graph: Graph) -> bool:
        """Save graph to database"""
        try:
            # Calculate TTL (7 days from now)
            ttl = int(time.time()) + (7 * 24 * 60 * 60)
            
            item = {
                'graph_id': graph.id,
                'topic': graph.topic,
                'nodes': [node.dict() for node in graph.nodes],
                'edges': [edge.dict() for edge in graph.edges],
                'metadata': graph.metadata.dict(),
                'user_interactions': graph.user_interactions.dict(),
                'created_at': graph.created_at.isoformat(),
                'last_modified': graph.last_modified.isoformat(),
                'version': graph.version,
                'ttl': ttl,
                'topic_hash': self._hash_topic(graph.topic),
                'date_partition': graph.created_at.strftime('%Y-%m-%d')
            }
            
            self.graphs_table.put_item(Item=item)
            return True
            
        except ClientError as e:
            logger.error(f"Failed to save graph {graph.id}: {e}")
            return False
    
    async def get_graph(self, graph_id: str) -> Optional[Graph]:
        """Retrieve graph by ID"""
        try:
            response = self.graphs_table.get_item(
                Key={'graph_id': graph_id}
            )
            
            if 'Item' not in response:
                return None
            
            item = response['Item']
            
            # Reconstruct graph object
            graph_data = {
                'id': item['graph_id'],
                'topic': item['topic'],
                'nodes': item['nodes'],
                'edges': item['edges'],
                'metadata': item['metadata'],
                'user_interactions': item['user_interactions'],
                'created_at': datetime.fromisoformat(item['created_at']),
                'last_modified': datetime.fromisoformat(item['last_modified']),
                'version': item['version']
            }
            
            return Graph.parse_obj(graph_data)
            
        except ClientError as e:
            logger.error(f"Failed to get graph {graph_id}: {e}")
            return None
    
    async def update_graph(
        self,
        graph_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update graph with partial data"""
        try:
            update_expression = "SET last_modified = :modified"
            expression_values = {
                ':modified': datetime.utcnow().isoformat()
            }
            
            # Build update expression
            for key, value in updates.items():
                if key == 'add_nodes':
                    update_expression += ", nodes = list_append(nodes, :new_nodes)"
                    expression_values[':new_nodes'] = [node.dict() for node in value]
                elif key == 'add_edges':
                    update_expression += ", edges = list_append(edges, :new_edges)"
                    expression_values[':new_edges'] = [edge.dict() for edge in value]
                elif key == 'increment_version':
                    update_expression += ", version = version + :inc"
                    expression_values[':inc'] = 1
                else:
                    update_expression += f", {key} = :{key}"
                    expression_values[f':{key}'] = value
            
            self.graphs_table.update_item(
                Key={'graph_id': graph_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values
            )
            
            return True
            
        except ClientError as e:
            logger.error(f"Failed to update graph {graph_id}: {e}")
            return False
    
    async def add_nodes_to_graph(
        self,
        graph_id: str,
        new_nodes: List[Node],
        new_edges: List[Edge]
    ) -> bool:
        """Add nodes and edges to existing graph"""
        return await self.update_graph(graph_id, {
            'add_nodes': new_nodes,
            'add_edges': new_edges,
            'increment_version': True
        })
    
    async def find_similar_graphs(
        self,
        topic: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find graphs with similar topics"""
        try:
            topic_hash = self._hash_topic(topic)
            
            response = self.graphs_table.query(
                IndexName='TopicIndex',
                KeyConditionExpression='topic_hash = :hash',
                ExpressionAttributeValues={':hash': topic_hash},
                ScanIndexForward=False,  # Most recent first
                Limit=limit
            )
            
            return response.get('Items', [])
            
        except ClientError as e:
            logger.error(f"Failed to find similar graphs: {e}")
            return []
    
    async def delete_graph(self, graph_id: str) -> bool:
        """Delete graph from database"""
        try:
            self.graphs_table.delete_item(
                Key={'graph_id': graph_id}
            )
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete graph {graph_id}: {e}")
            return False
    
    # Cache operations
    async def cache_set(
        self,
        key: str,
        value: Any,
        cache_type: str = "general",
        ttl_hours: int = 1
    ) -> bool:
        """Set cache value"""
        try:
            ttl = int(time.time()) + (ttl_hours * 3600)
            
            self.cache_table.put_item(
                Item={
                    'cache_key': key,
                    'cache_value': json.dumps(value) if not isinstance(value, str) else value,
                    'cache_type': cache_type,
                    'created_at': datetime.utcnow().isoformat(),
                    'ttl': ttl
                }
            )
            
            return True
            
        except ClientError as e:
            logger.error(f"Failed to set cache {key}: {e}")
            return False
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            response = self.cache_table.get_item(
                Key={'cache_key': key}
            )
            
            if 'Item' not in response:
                return None
            
            cache_value = response['Item']['cache_value']
            
            # Try to parse as JSON
            try:
                return json.loads(cache_value)
            except json.JSONDecodeError:
                return cache_value
                
        except ClientError as e:
            logger.error(f"Failed to get cache {key}: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Database health check"""
        try:
            # Simple table describe operation
            self.graphs_table.table_status
            return True
        except Exception:
            return False
    
    def _hash_topic(self, topic: str) -> str:
        """Create consistent hash for topic"""
        normalized = topic.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    async def get_analytics_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get analytics data for date range"""
        try:
            # Query by date partition
            current_date = start_date
            all_items = []
            
            while current_date <= end_date:
                date_key = current_date.strftime('%Y-%m-%d')
                
                response = self.graphs_table.query(
                    IndexName='TimeIndex',
                    KeyConditionExpression='date_partition = :date',
                    ExpressionAttributeValues={':date': date_key}
                )
                
                all_items.extend(response.get('Items', []))
                current_date += timedelta(days=1)
            
            # Aggregate analytics
            return {
                'total_graphs': len(all_items),
                'unique_topics': len(set(item.get('topic', '') for item in all_items)),
                'avg_nodes_per_graph': sum(len(item.get('nodes', [])) for item in all_items) / len(all_items) if all_items else 0,
                'most_popular_topics': self._get_popular_topics(all_items)
            }
            
        except ClientError as e:
            logger.error(f"Failed to get analytics: {e}")
            return {}
    
    def _get_popular_topics(self, items: List[Dict]) -> List[Dict[str, Any]]:
        """Get most popular topics from items"""
        topic_counts = {}
        
        for item in items:
            topic = item.get('topic', '')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return [
            {'topic': topic, 'count': count}
            for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
```

## 4. Data Validation & Integrity

### Input Validation (`src/utils/validators.py`)
```python
import re
from typing import Any, Dict, List
from pydantic import ValidationError

def validate_topic(topic: str) -> str:
    """Validate and sanitize topic input"""
    if not topic or not isinstance(topic, str):
        raise ValueError("Topic must be a non-empty string")
    
    # Clean whitespace
    topic = ' '.join(topic.split())
    
    if len(topic) < 2:
        raise ValueError("Topic must be at least 2 characters long")
    
    if len(topic) > 200:
        raise ValueError("Topic must be less than 200 characters")
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'data:',
        r'vbscript:',
        r'onload=',
        r'onerror='
    ]
    
    topic_lower = topic.lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, topic_lower):
            raise ValueError("Topic contains invalid content")
    
    return topic

def validate_graph_consistency(graph: Dict[str, Any]) -> List[str]:
    """Validate graph data consistency"""
    issues = []
    
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    
    if not nodes:
        issues.append("Graph must contain at least one node")
        return issues
    
    node_ids = {node.get('id') for node in nodes}
    
    # Check for duplicate node IDs
    node_id_list = [node.get('id') for node in nodes]
    if len(node_ids) != len(node_id_list):
        issues.append("Duplicate node IDs found")
    
    # Validate edges reference existing nodes
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        
        if source not in node_ids:
            issues.append(f"Edge references non-existent source node: {source}")
        
        if target not in node_ids:
            issues.append(f"Edge references non-existent target node: {target}")
    
    # Check for orphaned nodes (except root)
    connected_nodes = set()
    for edge in edges:
        connected_nodes.add(edge.get('source'))
        connected_nodes.add(edge.get('target'))
    
    root_nodes = [node for node in nodes if node.get('id') == 'root' or not node.get('parent_id')]
    
    if len(root_nodes) != 1:
        issues.append("Graph must have exactly one root node")
    
    for node in nodes:
        node_id = node.get('id')
        if node_id != 'root' and node_id not in connected_nodes and node.get('parent_id'):
            issues.append(f"Orphaned node found: {node_id}")
    
    return issues

def sanitize_export_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize data for export"""
    sanitized = data.copy()
    
    # Remove sensitive metadata
    if 'metadata' in sanitized:
        metadata = sanitized['metadata']
        sensitive_fields = ['ai_model_version', 'prompt_version', 'generation_time']
        for field in sensitive_fields:
            metadata.pop(field, None)
    
    # Round numeric values
    if 'nodes' in sanitized:
        for node in sanitized['nodes']:
            if 'position' in node:
                for axis in ['x', 'y', 'z']:
                    if axis in node['position']:
                        node['position'][axis] = round(node['position'][axis], 2)
            
            if 'size' in node:
                node['size'] = round(node['size'], 2)
    
    return sanitized
```

This data modeling and database design provides:

- **Scalability**: DynamoDB with proper indexing and partitioning
- **Performance**: Optimized access patterns and caching strategies  
- **Data Integrity**: Comprehensive validation and consistency checks
- **Flexibility**: Extensible schema design with metadata support
- **Analytics**: Built-in tracking for usage patterns and optimization
- **Cost Efficiency**: TTL-based cleanup and on-demand scaling