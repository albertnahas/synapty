import React, { useRef, useEffect, useCallback, useState } from 'react';
import { Node, Edge } from '../types';

interface MindMap2DProps {
  nodes: Node[];
  edges: Edge[];
  onNodeClick: (node: Node) => void;
  onNodeHover: (node: Node | null) => void;
  onNodeDrag?: (nodeId: string, newPosition: { x: number, y: number, z: number }) => void;
  selectedNodeId?: string;
}

const MindMap2D: React.FC<MindMap2DProps> = ({
  nodes,
  edges,
  onNodeClick,
  onNodeHover,
  onNodeDrag,
  selectedNodeId,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [draggedNodeId, setDraggedNodeId] = useState<string | null>(null);
  const [nodeDragStart, setNodeDragStart] = useState({ x: 0, y: 0 });

  // Convert 3D positions to 2D
  const get2DPosition = (node: Node) => {
    // Use x and z coordinates for 2D positioning, ignore y (height)
    // Scale and center the positions
    const scale = 0.8;
    const centerX = 400;
    const centerY = 300;
    
    return {
      x: centerX + node.position.x * scale,
      y: centerY + node.position.z * scale, // Use z for y in 2D
    };
  };

  // Convert 2D screen coordinates back to 3D world coordinates
  const screenTo3D = (screenX: number, screenY: number) => {
    const scale = 0.8;
    const centerX = 400;
    const centerY = 300;
    
    return {
      x: (screenX - centerX) / scale,
      y: 0, // Keep y at 0 for now
      z: (screenY - centerY) / scale,
    };
  };

  const getNodeColor = (node: Node) => {
    if (node.id === selectedNodeId) return '#ffd700'; // Gold for selected
    if (node.id === hoveredNodeId) return '#667eea'; // Blue for hovered
    
    // Color based on level
    switch (node.level) {
      case 0: return '#ff6b6b'; // Red for center
      case 1: return '#4ecdc4'; // Teal for level 1
      case 2: return '#45b7d1'; // Blue for level 2
      default: return '#96ceb4'; // Green for deeper levels
    }
  };

  const getNodeSize = (node: Node) => {
    switch (node.level) {
      case 0: return 20; // Largest for center
      case 1: return 15;
      case 2: return 12;
      default: return 10;
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.target === svgRef.current) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - transform.x, y: e.clientY - transform.y });
    }
  };

  const handleNodeMouseDown = (e: React.MouseEvent, node: Node) => {
    e.stopPropagation();
    setDraggedNodeId(node.id);
    const rect = svgRef.current?.getBoundingClientRect();
    if (rect) {
      const x = (e.clientX - rect.left - transform.x) / transform.scale;
      const y = (e.clientY - rect.top - transform.y) / transform.scale;
      setNodeDragStart({ x, y });
    }
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isDragging && !draggedNodeId) {
      setTransform(prev => ({
        ...prev,
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      }));
    } else if (draggedNodeId && onNodeDrag) {
      const rect = svgRef.current?.getBoundingClientRect();
      if (rect) {
        const x = (e.clientX - rect.left - transform.x) / transform.scale;
        const y = (e.clientY - rect.top - transform.y) / transform.scale;
        
        const worldPos = screenTo3D(x, y);
        onNodeDrag(draggedNodeId, worldPos);
      }
    }
  }, [isDragging, draggedNodeId, dragStart, transform, onNodeDrag]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setDraggedNodeId(null);
  }, []);

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setTransform(prev => ({
      ...prev,
      scale: Math.max(0.1, Math.min(3, prev.scale * delta))
    }));
  }, []);

  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    
    const svg = svgRef.current;
    if (svg) {
      svg.addEventListener('wheel', handleWheel);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      if (svg) {
        svg.removeEventListener('wheel', handleWheel);
      }
    };
  }, [handleMouseMove, handleMouseUp, handleWheel]);

  const handleNodeClick = (node: Node) => {
    onNodeClick(node);
  };

  const handleNodeMouseEnter = (node: Node) => {
    setHoveredNodeId(node.id);
    onNodeHover(node);
  };

  const handleNodeMouseLeave = () => {
    setHoveredNodeId(null);
    onNodeHover(null);
  };

  return (
    <div 
      style={{ 
        width: '100%', 
        height: '100%', 
        background: 'radial-gradient(circle, #1a1a2e 0%, #0f0f23 100%)',
        overflow: 'hidden',
        cursor: draggedNodeId ? 'grabbing' : isDragging ? 'grabbing' : 'grab'
      }}
    >
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        onMouseDown={handleMouseDown}
        style={{ display: 'block' }}
      >
        <defs>
          {/* Glow filter */}
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge> 
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          
          {/* Drop shadow filter */}
          <filter id="dropshadow">
            <feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.3"/>
          </filter>
        </defs>
        
        <g transform={`translate(${transform.x}, ${transform.y}) scale(${transform.scale})`}>
          {/* Render edges first (behind nodes) */}
          {edges.map((edge) => {
            const sourceNode = nodes.find(n => n.id === edge.source);
            const targetNode = nodes.find(n => n.id === edge.target);
            
            if (!sourceNode || !targetNode) return null;
            
            const sourcePos = get2DPosition(sourceNode);
            const targetPos = get2DPosition(targetNode);
            
            return (
              <line
                key={edge.id}
                x1={sourcePos.x}
                y1={sourcePos.y}
                x2={targetPos.x}
                y2={targetPos.y}
                stroke="rgba(255, 255, 255, 0.3)"
                strokeWidth="2"
                opacity="0.6"
              />
            );
          })}
          
          {/* Render nodes */}
          {nodes.map((node) => {
            const pos = get2DPosition(node);
            const radius = getNodeSize(node);
            const color = getNodeColor(node);
            const isHovered = node.id === hoveredNodeId;
            const isSelected = node.id === selectedNodeId;
            
            return (
              <g key={node.id}>
                {/* Glow effect */}
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={radius * 1.5}
                  fill={color}
                  opacity={isHovered || isSelected ? 0.2 : 0.1}
                  filter="url(#glow)"
                />
                
                {/* Main node circle */}
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={radius * (isHovered || isSelected ? 1.2 : 1)}
                  fill={color}
                  stroke={isSelected ? '#ffd700' : 'rgba(255, 255, 255, 0.3)'}
                  strokeWidth={isSelected ? 3 : 1}
                  filter="url(#dropshadow)"
                  style={{ cursor: draggedNodeId === node.id ? 'grabbing' : 'grab' }}
                  onMouseEnter={() => handleNodeMouseEnter(node)}
                  onMouseLeave={handleNodeMouseLeave}
                  onMouseDown={(e) => handleNodeMouseDown(e, node)}
                  onClick={() => !draggedNodeId && handleNodeClick(node)}
                />
                
                {/* Node label */}
                <text
                  x={pos.x}
                  y={pos.y - radius - 8}
                  textAnchor="middle"
                  fill="white"
                  fontSize={node.level === 0 ? 14 : 11}
                  fontWeight={node.level === 0 ? 'bold' : 'normal'}
                  filter="url(#dropshadow)"
                  style={{ 
                    pointerEvents: 'none', 
                    userSelect: 'none',
                    textShadow: '1px 1px 2px rgba(0,0,0,0.8)'
                  }}
                >
                  {node.title.length > 20 ? node.title.substring(0, 20) + '...' : node.title}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      
      {/* 2D Controls Info */}
      <div 
        style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          color: 'rgba(255, 255, 255, 0.6)',
          fontSize: '12px',
          pointerEvents: 'none'
        }}
      >
        Drag background to pan • Scroll to zoom • Drag nodes to reposition
      </div>
    </div>
  );
};

export default MindMap2D;