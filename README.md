# Synapty - AI-Powered Dynamic Mindmap Learning Tool

An interactive 3D mindmap visualization tool that transforms any topic into an explorable knowledge graph using AI.

## Features

- **Instant Graph Generation**: Type any topic and get a 3D mindmap in under 2 seconds
- **Interactive 3D Visualization**: Navigate through concepts with smooth pan, zoom, and rotate controls
- **Dynamic Node Expansion**: Click any node to explore sub-concepts with AI-generated content
- **Multiple Export Formats**: Export your mindmaps as JSON or PowerPoint presentations
- **Beautiful 3D Interface**: Immersive visualization with floating nodes and smooth animations

## Quick Start

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create environment file:
```bash
cp .env.example .env
```

4. Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
```

5. Start the FastAPI server:
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

1. Open your browser to `http://localhost:5173`
2. Enter any topic in the input field (e.g., "Machine Learning", "Ancient Rome", "Climate Change")
3. Click "Generate" to create your mindmap
4. Use mouse controls to navigate:
   - **Left click + drag**: Rotate view
   - **Right click + drag**: Pan
   - **Scroll wheel**: Zoom in/out
5. Click on any node to select it and view details
6. Click "Expand" to add sub-concepts to any node
7. Export your mindmap using the control buttons

## Architecture

- **Frontend**: React + Vite + Three.js + React Three Fiber
- **Backend**: FastAPI + OpenAI GPT-4 + Uvicorn
- **3D Rendering**: Three.js with React Three Fiber and Drei
- **AI Integration**: OpenAI Agents for concept generation and expansion

## API Endpoints

- `POST /api/generate-graph`: Generate initial mindmap for a topic
- `POST /api/expand-node`: Expand a node with sub-concepts
- `GET /api/health`: Health check endpoint

## Performance Targets

- Initial map generation: < 2 seconds
- Node expansion: < 1 second
- Smooth 60fps 3D interactions
- Responsive design for all screen sizes

## Roadmap

- [ ] RAG integration with Pinecone for grounded content
- [ ] Google Slides export integration
- [ ] User accounts and saved mindmaps
- [ ] Collaborative editing
- [ ] Mobile app version
- [ ] Voice input for topic generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details