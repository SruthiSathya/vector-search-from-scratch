# Vector Search From Scratch

This project is a semantic search engine for distributed systems research papers. The goal of this project is to explore vector searches for texts and extend it to multimodal data like images. 

### Current Features Supported
- PDF Loader - Ingests PDFs from a folder
- Chunking texts using logical splitting
- Generate embeddings
- Extract approximate searches using FAISS
- Query interface for relevant texts

### Planned Roadmap
#### Phase 1 - Working Baseline
- Fully functional vector search pipeline for PDF documents using FAISS
- Validate the retrival correctness by retriving top 3 releavent search results

#### Phase 2 - From the Scratch Implementation
#### Phase 2 - From the Scratch Implementation
- Replace FAISS with manual cosine similarity for text extraction
- Implement HNSW
- Benchmark performances, recall, latency, memory usage

#### Phase 3 - Multimodal Search
Extend this search engine for images, video and audio

## Steps to Run this project 

- Activate the Poetry environment
`poetry env activate`
- Run the main method
`poetry run python main.py`
- You will be prompted to enter a query, just ask any questions related to Distributed Systems, trafeoffs, terminologies.

