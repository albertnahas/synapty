# Synapti API Specifications

## API Overview

### Base Configuration
- **Base URL**: `https://api.synapti.app/v1`
- **Protocol**: HTTPS only
- **Authentication**: API Key (Header: `X-API-Key`)
- **Content-Type**: `application/json`
- **Rate Limiting**: 100 requests/minute per IP

## 1. Graph Generation API

### POST /mindmap/generate

Generate initial mindmap from topic input.

#### Request
```json
{
  "topic": "Machine Learning",
  "max_nodes": 8,
  "depth_level": 1,
  "style_preference": "academic|casual|visual",
  "language": "en"
}
```

#### Response (Success - 200)
```json
{
  "success": true,
  "data": {
    "graph_id": "uuid-4-string",
    "topic": "Machine Learning",
    "nodes": [
      {
        "id": "node_001",
        "title": "Supervised Learning",
        "summary": "Learning with labeled training data to predict outcomes",
        "position": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "color": "#4F46E5",
        "size": 1.2,
        "expandable": true,
        "connections": ["node_002", "node_003"]
      }
    ],
    "edges": [
      {
        "id": "edge_001",
        "source": "root",
        "target": "node_001",
        "relationship": "contains",
        "strength": 0.8
      }
    ],
    "metadata": {
      "generation_time": 1.23,
      "complexity_score": 0.7,
      "total_expandable_nodes": 6
    }
  }
}
```

#### Response (Error - 400)
```json
{
  "success": false,
  "error": {
    "code": "INVALID_TOPIC",
    "message": "Topic must be between 2 and 100 characters",
    "details": {
      "field": "topic",
      "value": "",
      "constraints": {
        "min_length": 2,
        "max_length": 100
      }
    }
  }
}
```

## 2. Node Expansion API

### POST /mindmap/{graph_id}/expand

Expand a specific node with sub-concepts.

#### Request
```json
{
  "node_id": "node_001",
  "expansion_depth": 1,
  "max_children": 5,
  "focus_areas": ["practical_applications", "key_concepts"]
}
```

#### Response (Success - 200)
```json
{
  "success": true,
  "data": {
    "parent_node_id": "node_001",
    "new_nodes": [
      {
        "id": "node_001_001",
        "title": "Linear Regression",
        "summary": "Predicts continuous values using linear relationships",
        "position": {
          "x": -2.5,
          "y": 1.0,
          "z": 0.5
        },
        "color": "#3B82F6",
        "size": 1.0,
        "expandable": true,
        "parent_id": "node_001"
      }
    ],
    "new_edges": [
      {
        "id": "edge_001_001",
        "source": "node_001",
        "target": "node_001_001",
        "relationship": "implements",
        "strength": 0.9
      }
    ],
    "layout_updates": [
      {
        "node_id": "node_002",
        "new_position": {
          "x": 2.0,
          "y": 1.0,
          "z": -0.5
        }
      }
    ]
  }
}
```

## 3. Graph Management API

### GET /mindmap/{graph_id}

Retrieve complete graph state.

#### Response (Success - 200)
```json
{
  "success": true,
  "data": {
    "graph_id": "uuid-4-string",
    "topic": "Machine Learning",
    "created_at": "2025-07-27T10:30:00Z",
    "last_modified": "2025-07-27T10:35:00Z",
    "nodes": [...],
    "edges": [...],
    "user_interactions": {
      "expansions_count": 3,
      "time_spent": 180,
      "last_activity": "2025-07-27T10:35:00Z"
    }
  }
}
```

### DELETE /mindmap/{graph_id}

Delete a mindmap session.

#### Response (Success - 204)
```json
{
  "success": true,
  "message": "Mindmap deleted successfully"
}
```

## 4. Export API

### POST /mindmap/{graph_id}/export

Export mindmap in various formats.

#### Request
```json
{
  "format": "json|pptx|google_slides",
  "options": {
    "include_metadata": true,
    "layout_style": "hierarchical|radial|force_directed",
    "theme": "light|dark|custom",
    "slide_template": "minimal|detailed|presentation"
  }
}
```

#### Response (JSON Export - 200)
```json
{
  "success": true,
  "data": {
    "download_url": "https://exports.synapti.app/{export_id}.json",
    "expires_at": "2025-07-27T11:30:00Z",
    "file_size": 15420,
    "format": "json"
  }
}
```

#### Response (PPTX Export - 200)
```json
{
  "success": true,
  "data": {
    "download_url": "https://exports.synapti.app/{export_id}.pptx",
    "expires_at": "2025-07-27T11:30:00Z",
    "file_size": 2456789,
    "format": "pptx",
    "slides_count": 12
  }
}
```

#### Response (Google Slides Export - 200)
```json
{
  "success": true,
  "data": {
    "google_slides_url": "https://docs.google.com/presentation/d/{presentation_id}",
    "presentation_id": "google-slides-id",
    "slides_count": 12,
    "sharing_settings": "private"
  }
}
```

## 5. Real-time Updates API

### WebSocket /ws/mindmap/{graph_id}

Real-time updates for collaborative features (Phase 2).

#### Messages
```json
// Incoming: Node expansion request
{
  "type": "expand_node",
  "data": {
    "node_id": "node_001",
    "user_id": "user_123"
  }
}

// Outgoing: Graph update
{
  "type": "graph_updated",
  "data": {
    "new_nodes": [...],
    "new_edges": [...],
    "updated_by": "user_123"
  }
}
```

## 6. Data Models

### Graph Schema
```typescript
interface Graph {
  id: string;
  topic: string;
  nodes: Node[];
  edges: Edge[];
  metadata: GraphMetadata;
  created_at: string;
  last_modified: string;
}

interface Node {
  id: string;
  title: string;
  summary: string;
  position: Position3D;
  color: string;
  size: number;
  expandable: boolean;
  parent_id?: string;
  connections: string[];
  metadata?: NodeMetadata;
}

interface Edge {
  id: string;
  source: string;
  target: string;
  relationship: 'contains' | 'implements' | 'relates_to' | 'depends_on';
  strength: number; // 0.0 - 1.0
  style?: EdgeStyle;
}

interface Position3D {
  x: number;
  y: number;
  z: number;
}

interface GraphMetadata {
  generation_time: number;
  complexity_score: number;
  total_expandable_nodes: number;
  ai_model_version: string;
}
```

## 7. Error Handling

### Standard Error Response
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      "field": "field_name",
      "value": "invalid_value",
      "constraints": {}
    },
    "request_id": "uuid-4-string",
    "timestamp": "2025-07-27T10:30:00Z"
  }
}
```

### Error Codes
- `INVALID_TOPIC`: Topic validation failed
- `GRAPH_NOT_FOUND`: Graph ID not found
- `NODE_NOT_FOUND`: Node ID not found
- `EXPANSION_FAILED`: AI expansion service failed
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `EXPORT_FAILED`: Export generation failed
- `INTERNAL_ERROR`: Unexpected server error

## 8. Performance Specifications

### SLA Targets
- **Graph Generation**: <2000ms (95th percentile)
- **Node Expansion**: <1000ms (95th percentile)
- **Export Generation**: <5000ms (95th percentile)
- **Cache Hit Response**: <100ms (99th percentile)

### Rate Limits
- **Anonymous**: 10 requests/minute
- **Authenticated**: 100 requests/minute
- **Premium**: 1000 requests/minute

### Caching Strategy
- **Graph Cache**: 1 hour TTL
- **Node Expansion**: 30 minutes TTL
- **Export Files**: 24 hours TTL
- **Static Assets**: 7 days TTL

## 9. OpenAI Integration Patterns

### Structured Output Schema
```json
{
  "mindmap_generation": {
    "type": "object",
    "properties": {
      "nodes": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "title": {"type": "string", "maxLength": 50},
            "summary": {"type": "string", "maxLength": 200},
            "category": {"type": "string"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1}
          },
          "required": ["title", "summary"]
        },
        "minItems": 4,
        "maxItems": 8
      }
    }
  }
}
```

### Prompt Engineering Templates
```python
MINDMAP_GENERATION_PROMPT = """
Create a mindmap for the topic: {topic}

Requirements:
- Generate {max_nodes} main concepts
- Each concept should have a clear, concise title (max 50 chars)
- Provide a summary for each concept (100-200 chars)
- Focus on {style_preference} style
- Ensure concepts are interconnected and logical

Output format: Use the provided JSON schema
"""

NODE_EXPANSION_PROMPT = """
Expand the concept "{node_title}" with sub-concepts.

Context: {parent_summary}
Focus areas: {focus_areas}

Requirements:
- Generate 3-5 sub-concepts
- Each sub-concept should be specific and actionable
- Maintain academic rigor while being accessible
- Show clear relationships to parent concept

Output format: Use the provided JSON schema
"""
```