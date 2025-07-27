# Synapti Frontend Architecture

## Architecture Overview

### Technology Stack
- **Framework**: React 18 + TypeScript 5.0+
- **Build Tool**: Vite 5.0+ (HMR, fast builds, optimized bundling)
- **3D Engine**: Three.js + React Three Fiber + Drei
- **State Management**: Zustand (lightweight, performant)
- **Styling**: Tailwind CSS + Styled Components
- **Animation**: Framer Motion + React Spring
- **Testing**: Vitest + React Testing Library + Playwright

## 1. Component Architecture

### Component Hierarchy
```
App
├── Layout
│   ├── Header
│   ├── Sidebar (export controls)
│   └── Footer
├── MindmapContainer
│   ├── TopicInput
│   ├── Canvas3D
│   │   ├── Scene
│   │   ├── NodeRenderer
│   │   ├── EdgeRenderer
│   │   ├── Camera
│   │   └── Controls
│   ├── NodeTooltip
│   ├── LoadingOverlay
│   └── ErrorBoundary
└── ExportPanel
    ├── FormatSelector
    ├── OptionsPanel
    └── ExportButton
```

### Core Components Design

#### 1. App Component (`src/App.tsx`)
```typescript
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Layout } from './components/Layout';
import { MindmapContainer } from './components/MindmapContainer';
import { useMindmapStore } from './stores/mindmapStore';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <Layout>
          <MindmapContainer />
        </Layout>
      </ErrorBoundary>
    </QueryClientProvider>
  );
};
```

#### 2. MindmapContainer (`src/components/MindmapContainer.tsx`)
```typescript
import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { TopicInput } from './TopicInput';
import { LoadingOverlay } from './LoadingOverlay';
import { NodeTooltip } from './NodeTooltip';
import { useMindmapStore } from '../stores/mindmapStore';
import { Scene3D } from './3d/Scene3D';

export const MindmapContainer: React.FC = () => {
  const { graph, isLoading, selectedNode } = useMindmapStore();

  return (
    <div className="relative h-screen w-full bg-gradient-to-br from-slate-900 to-slate-700">
      <TopicInput />
      
      <Suspense fallback={<LoadingOverlay />}>
        <Canvas
          camera={{ position: [0, 0, 10], fov: 60 }}
          gl={{ 
            antialias: true, 
            alpha: true,
            powerPreference: "high-performance"
          }}
          dpr={Math.min(window.devicePixelRatio, 2)}
        >
          <Scene3D graph={graph} />
        </Canvas>
      </Suspense>

      {selectedNode && <NodeTooltip node={selectedNode} />}
      {isLoading && <LoadingOverlay />}
    </div>
  );
};
```

#### 3. Scene3D (`src/components/3d/Scene3D.tsx`)
```typescript
import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { NodeRenderer } from './NodeRenderer';
import { EdgeRenderer } from './EdgeRenderer';
import { CameraController } from './CameraController';
import type { Graph } from '../types/graph';

interface Scene3DProps {
  graph: Graph | null;
}

export const Scene3D: React.FC<Scene3DProps> = ({ graph }) => {
  const groupRef = useRef<THREE.Group>();

  const { nodes, edges } = useMemo(() => {
    if (!graph) return { nodes: [], edges: [] };
    return {
      nodes: graph.nodes,
      edges: graph.edges
    };
  }, [graph]);

  // Smooth rotation animation
  useFrame((state, delta) => {
    if (groupRef.current && !state.controls?.enabled) {
      groupRef.current.rotation.y += delta * 0.1;
    }
  });

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight 
        position={[10, 10, 5]} 
        intensity={1} 
        castShadow
      />
      <pointLight position={[-10, -10, -5]} intensity={0.5} />

      {/* Environment */}
      <Environment preset="studio" />

      {/* Controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        zoomSpeed={0.6}
        panSpeed={0.8}
        rotateSpeed={0.4}
        minDistance={3}
        maxDistance={50}
      />

      <CameraController />

      {/* Graph Content */}
      <group ref={groupRef}>
        {/* Render Edges First (behind nodes) */}
        {edges.map((edge) => (
          <EdgeRenderer key={edge.id} edge={edge} nodes={nodes} />
        ))}

        {/* Render Nodes */}
        {nodes.map((node) => (
          <NodeRenderer key={node.id} node={node} />
        ))}
      </group>
    </>
  );
};
```

#### 4. NodeRenderer (`src/components/3d/NodeRenderer.tsx`)
```typescript
import React, { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, Sphere } from '@react-three/drei';
import { useMindmapStore } from '../../stores/mindmapStore';
import { useNodeExpansion } from '../../hooks/useNodeExpansion';
import type { Node } from '../../types/graph';

interface NodeRendererProps {
  node: Node;
}

export const NodeRenderer: React.FC<NodeRendererProps> = ({ node }) => {
  const meshRef = useRef<THREE.Mesh>();
  const [hovered, setHovered] = useState(false);
  const [clicked, setClicked] = useState(false);
  
  const { setSelectedNode, expandNode } = useMindmapStore();
  const { expandNodeMutation, isExpanding } = useNodeExpansion();

  // Animate on hover and selection
  useFrame((state, delta) => {
    if (meshRef.current) {
      const targetScale = hovered ? 1.2 : clicked ? 1.1 : 1.0;
      meshRef.current.scale.lerp(
        { x: targetScale, y: targetScale, z: targetScale } as any,
        delta * 5
      );
    }
  });

  const handleClick = async (event: any) => {
    event.stopPropagation();
    setClicked(true);
    setSelectedNode(node);

    if (node.expandable && !isExpanding) {
      try {
        await expandNodeMutation.mutateAsync(node.id);
      } catch (error) {
        console.error('Failed to expand node:', error);
      }
    }

    setTimeout(() => setClicked(false), 200);
  };

  const handlePointerOver = (event: any) => {
    event.stopPropagation();
    setHovered(true);
    document.body.style.cursor = 'pointer';
  };

  const handlePointerOut = () => {
    setHovered(false);
    document.body.style.cursor = 'auto';
  };

  return (
    <group position={[node.position.x, node.position.y, node.position.z]}>
      {/* Node Sphere */}
      <Sphere
        ref={meshRef}
        args={[node.size, 32, 32]}
        onClick={handleClick}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
      >
        <meshStandardMaterial
          color={node.color}
          metalness={0.1}
          roughness={0.2}
          transparent
          opacity={0.9}
        />
      </Sphere>

      {/* Node Label */}
      <Text
        position={[0, node.size + 0.5, 0]}
        fontSize={0.3}
        color="white"
        anchorX="center"
        anchorY="middle"
        maxWidth={3}
        textAlign="center"
      >
        {node.title}
      </Text>

      {/* Expansion Indicator */}
      {node.expandable && (
        <Text
          position={[0, -node.size - 0.3, 0]}
          fontSize={0.2}
          color="#10B981"
          anchorX="center"
          anchorY="middle"
        >
          {isExpanding ? "..." : "+"}
        </Text>
      )}

      {/* Loading Indicator */}
      {isExpanding && (
        <Sphere args={[node.size + 0.1, 16, 16]}>
          <meshBasicMaterial
            color="#3B82F6"
            transparent
            opacity={0.3}
            wireframe
          />
        </Sphere>
      )}
    </group>
  );
};
```

## 2. State Management Architecture

### Zustand Store Design (`src/stores/mindmapStore.ts`)
```typescript
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { Graph, Node, Edge } from '../types/graph';

interface MindmapState {
  // Graph State
  graph: Graph | null;
  selectedNode: Node | null;
  hoveredNode: Node | null;
  
  // UI State
  isLoading: boolean;
  isGenerating: boolean;
  isExpanding: boolean;
  error: string | null;
  
  // Export State
  exportFormat: 'json' | 'pptx' | 'google_slides';
  isExporting: boolean;
  
  // Session State
  currentTopic: string;
  sessionId: string | null;
  expandedNodes: Set<string>;
}

interface MindmapActions {
  // Graph Actions
  setGraph: (graph: Graph) => void;
  updateGraph: (updates: Partial<Graph>) => void;
  addNodes: (nodes: Node[], edges: Edge[]) => void;
  
  // Node Actions
  setSelectedNode: (node: Node | null) => void;
  setHoveredNode: (node: Node | null) => void;
  expandNode: (nodeId: string) => void;
  
  // UI Actions
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  // Export Actions
  setExportFormat: (format: 'json' | 'pptx' | 'google_slides') => void;
  setExporting: (exporting: boolean) => void;
  
  // Session Actions
  startNewSession: (topic: string) => void;
  clearSession: () => void;
}

type MindmapStore = MindmapState & MindmapActions;

export const useMindmapStore = create<MindmapStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial State
        graph: null,
        selectedNode: null,
        hoveredNode: null,
        isLoading: false,
        isGenerating: false,
        isExpanding: false,
        error: null,
        exportFormat: 'json',
        isExporting: false,
        currentTopic: '',
        sessionId: null,
        expandedNodes: new Set(),

        // Actions
        setGraph: (graph) => set({ graph, isLoading: false, error: null }),
        
        updateGraph: (updates) => set((state) => ({
          graph: state.graph ? { ...state.graph, ...updates } : null
        })),

        addNodes: (nodes, edges) => set((state) => {
          if (!state.graph) return state;
          
          return {
            graph: {
              ...state.graph,
              nodes: [...state.graph.nodes, ...nodes],
              edges: [...state.graph.edges, ...edges]
            },
            isExpanding: false
          };
        }),

        setSelectedNode: (node) => set({ selectedNode: node }),
        setHoveredNode: (node) => set({ hoveredNode: node }),
        
        expandNode: (nodeId) => set((state) => ({
          expandedNodes: new Set([...state.expandedNodes, nodeId]),
          isExpanding: true
        })),

        setLoading: (loading) => set({ isLoading: loading }),
        setError: (error) => set({ error, isLoading: false }),

        setExportFormat: (format) => set({ exportFormat: format }),
        setExporting: (exporting) => set({ isExporting: exporting }),

        startNewSession: (topic) => set({
          currentTopic: topic,
          sessionId: crypto.randomUUID(),
          graph: null,
          selectedNode: null,
          expandedNodes: new Set(),
          error: null
        }),

        clearSession: () => set({
          graph: null,
          selectedNode: null,
          hoveredNode: null,
          currentTopic: '',
          sessionId: null,
          expandedNodes: new Set(),
          error: null
        })
      }),
      {
        name: 'mindmap-storage',
        partialize: (state) => ({
          currentTopic: state.currentTopic,
          exportFormat: state.exportFormat
        })
      }
    ),
    { name: 'mindmap-store' }
  )
);
```

## 3. Custom Hooks Architecture

### API Integration Hooks (`src/hooks/useApi.ts`)
```typescript
import { useMutation, useQuery } from '@tanstack/react-query';
import { mindmapApi } from '../services/api';
import { useMindmapStore } from '../stores/mindmapStore';
import type { GenerateGraphRequest, ExpandNodeRequest } from '../types/api';

export const useGraphGeneration = () => {
  const { setGraph, setLoading, setError } = useMindmapStore();

  return useMutation({
    mutationFn: (request: GenerateGraphRequest) => 
      mindmapApi.generateGraph(request),
    
    onMutate: () => {
      setLoading(true);
      setError(null);
    },
    
    onSuccess: (data) => {
      setGraph(data.graph);
    },
    
    onError: (error: any) => {
      setError(error.message || 'Failed to generate mindmap');
      setLoading(false);
    }
  });
};

export const useNodeExpansion = () => {
  const { addNodes, setError } = useMindmapStore();

  return {
    expandNodeMutation: useMutation({
      mutationFn: ({ graphId, nodeId, options }: ExpandNodeRequest) =>
        mindmapApi.expandNode(graphId, nodeId, options),
      
      onSuccess: (data) => {
        addNodes(data.new_nodes, data.new_edges);
      },
      
      onError: (error: any) => {
        setError(error.message || 'Failed to expand node');
      }
    }),
    
    isExpanding: false // Will be managed by mutation state
  };
};

export const useGraphExport = () => {
  const { setExporting, setError } = useMindmapStore();

  return useMutation({
    mutationFn: ({ graphId, format, options }: ExportRequest) =>
      mindmapApi.exportGraph(graphId, format, options),
    
    onMutate: () => {
      setExporting(true);
      setError(null);
    },
    
    onSuccess: (data) => {
      // Handle download or redirect based on format
      if (data.download_url) {
        window.open(data.download_url, '_blank');
      } else if (data.google_slides_url) {
        window.open(data.google_slides_url, '_blank');
      }
      setExporting(false);
    },
    
    onError: (error: any) => {
      setError(error.message || 'Export failed');
      setExporting(false);
    }
  });
};
```

### Performance Hooks (`src/hooks/usePerformance.ts`)
```typescript
import { useCallback, useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useMindmapStore } from '../stores/mindmapStore';

export const usePerformanceOptimization = () => {
  const frameCount = useRef(0);
  const lastFpsUpdate = useRef(Date.now());
  const fps = useRef(60);

  useFrame(() => {
    frameCount.current++;
    const now = Date.now();
    
    if (now - lastFpsUpdate.current >= 1000) {
      fps.current = frameCount.current;
      frameCount.current = 0;
      lastFpsUpdate.current = now;
      
      // Adjust quality based on FPS
      if (fps.current < 30) {
        // Reduce quality
        document.querySelector('canvas')?.setAttribute('data-quality', 'low');
      } else if (fps.current > 50) {
        // Increase quality
        document.querySelector('canvas')?.setAttribute('data-quality', 'high');
      }
    }
  });

  return {
    fps: fps.current,
    isHighPerformance: fps.current > 50,
    isLowPerformance: fps.current < 30
  };
};

export const useMemoizedGraph = () => {
  const { graph } = useMindmapStore();
  
  return useMemo(() => {
    if (!graph) return { nodes: [], edges: [], positions: new Map() };
    
    const positions = new Map();
    graph.nodes.forEach(node => {
      positions.set(node.id, [node.position.x, node.position.y, node.position.z]);
    });
    
    return {
      nodes: graph.nodes,
      edges: graph.edges,
      positions
    };
  }, [graph]);
};
```

## 4. Performance Optimizations

### React Optimization Patterns
```typescript
// Memoized Components
export const NodeRenderer = React.memo<NodeRendererProps>(({ node }) => {
  // Component implementation
}, (prevProps, nextProps) => {
  return (
    prevProps.node.id === nextProps.node.id &&
    prevProps.node.position === nextProps.node.position &&
    prevProps.node.color === nextProps.node.color
  );
});

// Virtualized Rendering for Large Graphs
export const useVirtualizedNodes = (nodes: Node[], camera: Camera) => {
  return useMemo(() => {
    const frustum = new THREE.Frustum();
    const matrix = new THREE.Matrix4().multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse
    );
    frustum.setFromProjectionMatrix(matrix);
    
    return nodes.filter(node => {
      const sphere = new THREE.Sphere(
        new THREE.Vector3(node.position.x, node.position.y, node.position.z),
        node.size
      );
      return frustum.intersectsSphere(sphere);
    });
  }, [nodes, camera.position, camera.rotation]);
};
```

### Three.js Optimizations
```typescript
// Instanced Rendering for Multiple Similar Nodes
export const InstancedNodeRenderer: React.FC = () => {
  const meshRef = useRef<THREE.InstancedMesh>();
  const { nodes } = useMindmapStore();
  
  useEffect(() => {
    if (!meshRef.current) return;
    
    const tempObject = new THREE.Object3D();
    const tempColor = new THREE.Color();
    
    nodes.forEach((node, index) => {
      tempObject.position.set(node.position.x, node.position.y, node.position.z);
      tempObject.scale.setScalar(node.size);
      tempObject.updateMatrix();
      
      meshRef.current!.setMatrixAt(index, tempObject.matrix);
      meshRef.current!.setColorAt(index, tempColor.set(node.color));
    });
    
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [nodes]);
  
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, nodes.length]}>
      <sphereGeometry args={[1, 16, 16]} />
      <meshStandardMaterial />
    </instancedMesh>
  );
};
```

## 5. Responsive Design Architecture

### Breakpoint System
```typescript
export const breakpoints = {
  mobile: '(max-width: 768px)',
  tablet: '(max-width: 1024px)',
  desktop: '(min-width: 1025px)',
  touch: '(pointer: coarse)'
} as const;

export const useResponsive = () => {
  const [isMobile, setIsMobile] = useState(false);
  const [isTablet, setIsTablet] = useState(false);
  const [isTouch, setIsTouch] = useState(false);
  
  useEffect(() => {
    const mobileQuery = window.matchMedia(breakpoints.mobile);
    const tabletQuery = window.matchMedia(breakpoints.tablet);
    const touchQuery = window.matchMedia(breakpoints.touch);
    
    const updateMatches = () => {
      setIsMobile(mobileQuery.matches);
      setIsTablet(tabletQuery.matches);
      setIsTouch(touchQuery.matches);
    };
    
    updateMatches();
    
    mobileQuery.addEventListener('change', updateMatches);
    tabletQuery.addEventListener('change', updateMatches);
    touchQuery.addEventListener('change', updateMatches);
    
    return () => {
      mobileQuery.removeEventListener('change', updateMatches);
      tabletQuery.removeEventListener('change', updateMatches);
      touchQuery.removeEventListener('change', updateMatches);
    };
  }, []);
  
  return { isMobile, isTablet, isTouch, isDesktop: !isMobile && !isTablet };
};
```

### Adaptive Controls
```typescript
export const AdaptiveControls: React.FC = () => {
  const { isMobile, isTablet, isTouch } = useResponsive();
  
  if (isMobile) {
    return (
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        zoomSpeed={1.2}
        panSpeed={1.5}
        rotateSpeed={0.8}
        touches={{
          ONE: THREE.TOUCH.ROTATE,
          TWO: THREE.TOUCH.DOLLY_PAN
        }}
      />
    );
  }
  
  return (
    <OrbitControls
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
      zoomSpeed={0.6}
      panSpeed={0.8}
      rotateSpeed={0.4}
    />
  );
};
```

## 6. Testing Architecture

### Component Testing Setup
```typescript
// src/test-utils/setup.ts
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactElement } from 'react';

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false }
  }
});

const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <QueryClientProvider client={createTestQueryClient()}>
      {children}
    </QueryClientProvider>
  );
};

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options });

export * from '@testing-library/react';
export { customRender as render };
```

### 3D Component Testing
```typescript
// Mock Three.js for testing
jest.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => <div data-testid="canvas">{children}</div>,
  useFrame: jest.fn(),
  useThree: () => ({
    camera: { position: { x: 0, y: 0, z: 10 } },
    scene: {},
    gl: {}
  })
}));

// Example test
describe('NodeRenderer', () => {
  const mockNode: Node = {
    id: 'test-node',
    title: 'Test Node',
    summary: 'Test summary',
    position: { x: 0, y: 0, z: 0 },
    color: '#4F46E5',
    size: 1.0,
    expandable: true,
    connections: []
  };

  it('renders node with correct title', () => {
    render(<NodeRenderer node={mockNode} />);
    expect(screen.getByText('Test Node')).toBeInTheDocument();
  });

  it('handles node expansion on click', async () => {
    const mockExpand = jest.fn();
    jest.mocked(useMindmapStore).mockReturnValue({
      expandNode: mockExpand,
      // ... other store values
    });

    render(<NodeRenderer node={mockNode} />);
    
    const nodeElement = screen.getByRole('button');
    fireEvent.click(nodeElement);
    
    expect(mockExpand).toHaveBeenCalledWith('test-node');
  });
});
```

This frontend architecture provides:
- **Performance**: Optimized Three.js rendering with virtualization
- **Scalability**: Component-based architecture with proper state management
- **Responsiveness**: Adaptive controls and layouts for all devices
- **Maintainability**: TypeScript, proper separation of concerns, comprehensive testing
- **User Experience**: Smooth animations, intuitive interactions, error handling