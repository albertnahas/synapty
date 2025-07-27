Product Requirements Document (PRD)
Synapti
Project: AI-Powered Dynamic Mindmap Learning Tool
Version: 0.1 (MVP)
Date: 27 July 2025

⸻

1. Purpose & Vision

Enable learners of all backgrounds to type any topic and instantly receive an interactive, story-driven mindmap (“bubble map”) that visualizes key concepts, supports drilling into sub-topics on the fly, and exports seamlessly for sharing or presentation.

⸻

2. Objectives
   • Speed: Initial map generation in < 2 seconds.
   • Interactivity: Expand any node on click with sub-nodes loaded dynamically in < 1 second.
   • Clarity: Provide bite-sized, human-readable summaries within each bubble.
   • Shareability: One-click export of the current graph to Google Slides, PowerPoint (PPTX), or JSON.

⸻

3. User Personas

Persona Goals Pain Points
Student (18-24) Grasp new concepts quickly; study visually Overwhelmed by text; boring outlines
Lifelong Learner Explore unfamiliar topics; personalize learning Lack of structure; information overload
Educator/Tutor Create engaging teaching aids; adapt to levels Time-consuming slide prep; static charts

⸻

4. Key Features (MVP)
   1. Topic Input
      • Text field with “Generate” button.
   2. Graph Generation
      • FastAPI (Uvicorn) + OpenAI Agents backend returns JSON graph of 6–8 root nodes.
   3. Interactive Canvas
      • Frontend built with Vite, React, and Three.js:
      • Render nodes as spheres, edges as lines.
      • Enable 3D camera controls (pan, zoom, rotate).
   4. Node Expansion
      • Hover on node shows summary snippet and “+ Expand” button.
      • On click, fetch sub-concepts and merge into scene without full reload.
   5. Export Capability
      • Download current graph as JSON.
      • Export to PPTX via pptxgenjs integration.
      • (Optional) One-click “Export to Google Slides” link.

⸻

5. Functional Requirements
   • FR1: Accept a topic string and return a valid graph JSON schema.
   • FR2: Render nodes and edges with Three.js; support smooth 3D controls.
   • FR3: Handle node expansion requests & dynamically integrate new nodes.
   • FR4: Provide export endpoints for JSON and PPTX formats.

⸻

6. Non-Functional Requirements
   • NFR1 (Performance):
   • Initial map ≤ 2 s.
   • Node expansion ≤ 1 s under normal load.
   • NFR2 (Scalability):
   • Serverless backend (AWS Lambda) auto-scales per request.
   • NFR3 (Reliability):
   • 99% uptime SLA; no cold-start spin-down during active sessions.
   • NFR4 (Security):
   • CORS restricted to approved frontend domain(s).
   • OpenAI API key stored securely; never exposed client-side.

⸻

7. Technical Architecture

[ React/Vite/Three.js Frontend ]
↓ HTTPS
[ AWS API Gateway → Lambda (FastAPI + Mangum) → OpenAI Agents ]
↘ Optional RAG layer → Pinecone

    •	Frontend:
    •	Vite for fast builds.
    •	React for UI.
    •	Three.js for 3D rendering.
    •	Backend:
    •	AWS API Gateway + Lambda running FastAPI (Mangum).
    •	OpenAI Agents orchestrate concept extraction & summaries.
    •	Phase 2: Add RAG grounding via Pinecone vector store.

⸻

8. Success Metrics
   • Adoption: 100 beta sign-ups in week 1.
   • Engagement: Avg. session with > 3 node expansions.
   • Speed Compliance: 90% of requests meet performance SLAs.
   • Usability: ≥ 4 / 5 average rating from pilot testers.

⸻

9. Timeline & Milestones

Week Deliverable
1 Repo setup & “type topic → JSON graph” API
2 React/Three.js viewer + node expansion functionality
3 Export pipelines (JSON, PPTX)
4 Internal testing, UX polish
5 Beta launch + user feedback gathering

⸻

10. Risks & Mitigations
    • Hallucinations → Mitigation: Phase 2 RAG grounding against reliable sources.
    • Graph Complexity → Mitigation: Auto-cluster dense branches; implement lazy loading.
    • Lambda Cold Starts → Mitigation: Minimal dependencies; consider provisioned concurrency.

⸻

11. Next Steps
    1.  Finalize MVP backlog in Jira or Trello.
    2.  Kick off weekend hackathon sprint with core team.
    3.  Recruit 5–10 pilot users (students & educators) for early feedback.

⸻

With this roadmap in place, we’re ready to build an MVP that delivers AI-powered, interactive study bubbles—and get learners excited about visualizing knowledge like never before.
