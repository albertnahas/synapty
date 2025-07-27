import axios from 'axios';
import { GraphData, ExpandNodeRequest, ExpandNodeResponse } from './types';


const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  async generateGraph(topic: string): Promise<GraphData> {
    const response = await api.post('/api/generate-graph', { topic });
    return response.data;
  },

  async expandNode(nodeId: string, topic: string): Promise<ExpandNodeResponse> {
    const response = await api.post('/api/expand-node', {
      node_id: nodeId,
      topic: topic,
    });
    return response.data;
  },

  async healthCheck(): Promise<{ status: string; version: string }> {
    const response = await api.get('/api/health');
    return response.data;
  },
};