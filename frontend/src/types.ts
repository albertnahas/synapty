export interface Node {
  id: string;
  title: string;
  summary: string;
  position: {
    x: number;
    y: number;
    z: number;
  };
  children: string[];
  parent?: string;
  level: number;
}

export interface Edge {
  id: string;
  source: string;
  target: string;
}

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
  center_topic: string;
}

export interface ExpandNodeRequest {
  node_id: string;
  topic: string;
}

export interface ExpandNodeResponse {
  nodes: Node[];
  edges: Edge[];
  expanded_node: string;
}