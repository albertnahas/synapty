import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';
import { Node, Edge } from '../types';

interface NodeMeshProps {
  node: Node;
  onNodeClick: (node: Node) => void;
  onNodeHover: (node: Node | null) => void;
  isSelected: boolean;
}

const NodeMesh: React.FC<NodeMeshProps> = ({ node, onNodeClick, onNodeHover, isSelected }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      // Gentle floating animation
      meshRef.current.position.y = node.position.y + Math.sin(state.clock.elapsedTime + node.position.x * 0.01) * 2;
      
      // Gentle rotation
      meshRef.current.rotation.y += 0.002;
      
      // Scale animation on hover
      const targetScale = hovered || isSelected ? 1.2 : 1.0;
      meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });

  const getNodeColor = () => {
    if (isSelected) return '#ffd700'; // Gold for selected
    if (hovered) return '#667eea'; // Blue for hovered
    
    // Color based on level
    switch (node.level) {
      case 0: return '#ff6b6b'; // Red for center
      case 1: return '#4ecdc4'; // Teal for level 1
      case 2: return '#45b7d1'; // Blue for level 2
      default: return '#96ceb4'; // Green for deeper levels
    }
  };

  const getNodeSize = () => {
    switch (node.level) {
      case 0: return 25; // Largest for center
      case 1: return 18;
      case 2: return 15;
      default: return 12;
    }
  };

  return (
    <group position={[node.position.x, node.position.y, node.position.z]}>
      {/* Node sphere */}
      <Sphere
        ref={meshRef}
        args={[getNodeSize(), 32, 32]}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          onNodeHover(node);
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHovered(false);
          onNodeHover(null);
        }}
        onClick={(e) => {
          e.stopPropagation();
          onNodeClick(node);
        }}
      >
        <meshStandardMaterial
          color={getNodeColor()}
          emissive={getNodeColor()}
          emissiveIntensity={0.1}
          roughness={0.4}
          metalness={0.1}
        />
      </Sphere>
      
      {/* Node label */}
      <Text
        position={[0, getNodeSize() + 15, 0]}
        fontSize={node.level === 0 ? 16 : 12}
        color="white"
        anchorX="center"
        anchorY="middle"
        maxWidth={100}
      >
        {node.title}
      </Text>
      
      {/* Glow effect */}
      <Sphere args={[getNodeSize() * 1.5, 16, 16]}>
        <meshBasicMaterial
          color={getNodeColor()}
          transparent
          opacity={hovered || isSelected ? 0.1 : 0.05}
        />
      </Sphere>
    </group>
  );
};

interface EdgeLineProps {
  edge: Edge;
  nodes: Node[];
}

const EdgeLine: React.FC<EdgeLineProps> = ({ edge, nodes }) => {
  const sourceNode = nodes.find(n => n.id === edge.source);
  const targetNode = nodes.find(n => n.id === edge.target);

  if (!sourceNode || !targetNode) return null;

  const points = [
    new THREE.Vector3(sourceNode.position.x, sourceNode.position.y, sourceNode.position.z),
    new THREE.Vector3(targetNode.position.x, targetNode.position.y, targetNode.position.z),
  ];

  return (
    <Line
      points={points}
      color="rgba(255, 255, 255, 0.3)"
      lineWidth={2}
      transparent
      opacity={0.6}
    />
  );
};

interface MindMapVisualizationProps {
  nodes: Node[];
  edges: Edge[];
  onNodeClick: (node: Node) => void;
  onNodeHover: (node: Node | null) => void;
  selectedNodeId?: string;
}

const MindMapVisualization: React.FC<MindMapVisualizationProps> = ({
  nodes,
  edges,
  onNodeClick,
  onNodeHover,
  selectedNodeId,
}) => {
  return (
    <Canvas
      camera={{ position: [0, 200, 500], fov: 60 }}
      style={{ 
        width: '100%', 
        height: '100%', 
        background: 'radial-gradient(circle, #1a1a2e 0%, #0f0f23 100%)' 
      }}
    >
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[100, 100, 100]} intensity={0.8} />
      <pointLight position={[-100, -100, -100]} intensity={0.4} color="#667eea" />
      
      {/* Camera controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={200}
        maxDistance={1000}
        autoRotate={false}
        autoRotateSpeed={0.5}
      />
      
      {/* Edges */}
      {edges.map((edge) => (
        <EdgeLine key={edge.id} edge={edge} nodes={nodes} />
      ))}
      
      {/* Nodes */}
      {nodes.map((node) => (
        <NodeMesh
          key={node.id}
          node={node}
          onNodeClick={onNodeClick}
          onNodeHover={onNodeHover}
          isSelected={node.id === selectedNodeId}
        />
      ))}
      
      {/* Background stars */}
      <mesh>
        <sphereGeometry args={[1000, 32, 32]} />
        <meshBasicMaterial color="#000011" side={THREE.BackSide} />
      </mesh>
    </Canvas>
  );
};

export default MindMapVisualization;