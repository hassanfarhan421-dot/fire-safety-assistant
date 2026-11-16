# Fire Safety Assistant

A multimodal Retrieval-Augmented Generation (RAG) system for NASA fire and microgravity research. The application combines text, image, and experimental data retrieval with Gemini to provide evidence-based research insights through a FastAPI web application.

## Overview

Understanding fire behavior in microgravity is important for the safety of future human spaceflight missions.

This project provides a research assistant capable of working with multiple types of scientific data instead of relying only on traditional text-based RAG.

It can process:

- 📄 Research documents and PDFs
- 🖼️ Scientific images
- 📊 CSV and experimental datasets
- 🔎 Semantically retrieved research information
- 📈 Experimental data and visualizations
- 🤖 Gemini-powered research analysis

The retrieved information is passed to Gemini as context, allowing the system to generate structured answers based on the available research data.

## Architecture

```text
                    Research Data
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Documents       Images        CSV Tables
          │              │              │
          ▼              ▼              ▼
   Text Embeddings      CLIP       Row Processing
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                     ChromaDB
                         │
                         ▼
                    User Question
                         │
                         ▼
                 Relevant Retrieval
                         │
                         ▼
                       Gemini
                         │
                         ▼
                Structured Response
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
     Findings         Evidence          Charts
```

## Features

### Multimodal Retrieval

Different retrieval methods are used depending on the data:

- **Text** — Hugging Face embeddings
- **Images** — CLIP image/text embeddings
- **Tables** — CSV rows converted into searchable documents
- **Vector Storage** — ChromaDB

### Research Analysis

The assistant can return:

- Key research findings
- Experimental measurements
- Comparisons between experimental conditions
- Relevant source information
- Related research images
- Space-safety implications
- Limitations
- Data visualizations

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| FastAPI | Backend and API |
| LangChain | RAG pipeline |
| Gemini | LLM analysis and generation |
| ChromaDB | Vector database |
| Hugging Face | Text embeddings |
| CLIP | Image embeddings |
| Pandas | Experimental data processing |
| PyTorch | Image model inference |
| Pydantic | Structured responses |

## Project Structure

```text
fire-safety-assistant/
│
├── fire-safety-assistant.py
├── requirements.txt
├── DockerFile
├── .dockerignore
├── .gitignore
│
├── static/
│   └── index.html
│
├── uploaded_images/
│
└── chromadb/
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hassanfarhan421-dot/fire-safety-assistant.git
cd fire-safety-assistant
```

### 2. Create a virtual environment

```bash
py -3.13 -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
py -3.13 -m pip install -r requirements.txt
```

### 4. Configure Gemini

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_api_key_here
```

Keep your API key private and do not commit `.env` to GitHub.

### 5. Run the application

```bash
py -3.13 -m uvicorn fire-safety-assistant:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

## Data Workflow

1. Upload research documents, images, or experimental datasets.
2. The application processes the uploaded data.
3. Embeddings are generated for searchable content.
4. ChromaDB stores the resulting vectors.
5. A user submits a research question.
6. Relevant information is retrieved from the vector database.
7. Retrieved evidence is provided to Gemini.
8. Gemini generates a structured research response.
9. Relevant findings, measurements, images, and visualizations are returned to the user.

## Example Questions

```text
How does oxygen concentration affect flame temperature
under different gravity conditions?

Compare the experimental flame spread rates between
different fuel materials.

What fire-safety implications can be identified from
the retrieved microgravity experiments?
```

## Docker

The repository includes a `DockerFile` for containerizing the application.

Build the image:

```bash
docker build -t fire-safety-assistant .
```

Run it:

```bash
docker run --env-file .env -p 8000:8000 fire-safety-assistant
```

The application will then be available at:

```text
http://localhost:8000
```

## Research Scope

This project is designed as a research and educational prototype for exploring fire behavior and safety information related to microgravity environments.

It is **not** an official NASA system and its generated responses should not be treated as authoritative NASA findings or as a substitute for professional flight-safety analysis.

Any synthetic datasets used for testing are test data and should not be interpreted as real NASA experimental results.

## Future Development

Potential improvements include:

- Larger-scale multimodal datasets
- Improved metadata filtering
- More advanced multimodal retrieval
- Better source and citation tracking
- Additional scientific visualizations
- Persistent cloud storage
- Authentication and user management
- Production deployment
- Automated ingestion pipelines

## License

This project is intended for educational and research purposes.