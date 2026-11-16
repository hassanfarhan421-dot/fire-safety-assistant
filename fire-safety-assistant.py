# imports
from langchain_community.document_loaders import UnstructuredPDFLoader
import matplotlib.pyplot as plt
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import base64
import os
from fastapi.responses import RedirectResponse
import io
import uuid
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import uuid
from langchain_chroma import Chroma
from typing import Literal, List
from datetime import date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from PIL import Image
import clip
import chromadb
import uuid
from pathlib import Path
import torch
from PIL import Image, UnidentifiedImageError
from chromadb.utils.embedding_functions import EmbeddingFunction
import torch
import torch.nn.functional as f
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from fastapi import File
from langchain_core.documents import Document
import json


class ExperimentalMeasurement(BaseModel):
    experiment_id: str
    fuel: Optional[str] = None
    gravity_condition: Optional[str] = None
    measurement: str
    value: Optional[float] = None
    unit: Optional[str] = None


class Charts(BaseModel):
    chart_type: Literal["bar", "line", "pie"]
    title: str
    labels: list[str]
    values: list[float]


class Evidence(BaseModel):
    finding: str
    evidence: str
    source: str
    page: Optional[int] = None


class RelatedImage(BaseModel):
    image_id: str
    description: Optional[str] = None
    source: str
    page: Optional[int] = None
    image_url: Optional[str] = None


class FireSafetyResponse(BaseModel):
    answer: str
    charts: Optional[Charts] = None
    key_findings: list[Evidence]
    experimental_data: list[ExperimentalMeasurement]
    related_images: list[RelatedImage]
    space_safety_implications: list[str]
    limitations: list[str]
    sources: list[Evidence]


class ChatRequest(BaseModel):
    question: str


app = FastAPI()


@app.get("/")
def root():
    return RedirectResponse("/ui/")


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / 'uploaded_pdfs'

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_UPLOAD_DIR = BASE_DIR / 'uploaded_tables'

TABLE_UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CHROMA_PATH = BASE_DIR / 'chromadb'

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

IMAGE_UPLOAD_DIR = BASE_DIR / "uploaded_images"
IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/plots",
    StaticFiles(directory=str(PLOTS_DIR)),
    name="plots"
)

app.mount(
    "/ui",
    StaticFiles(directory=str(STATIC_DIR), html=True),
    name="frontend"
)

app.mount(
    "/uploaded_images",
    StaticFiles(directory=str(IMAGE_UPLOAD_DIR)),
    name="uploaded_images"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
    max_output_tokens=4096
)
structured_llm = llm.with_structured_output(
    FireSafetyResponse, method="json_schema")

text_embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100)

text_collection = Chroma(
    collection_name='text',
    embedding_function=text_embeddings,
    persist_directory=str(CHROMA_PATH)
)

table_collection = Chroma(
    collection_name="tables",
    embedding_function=text_embeddings,
    persist_directory=str(CHROMA_PATH)
)

client = chromadb.PersistentClient(path=str(CHROMA_PATH))


# MultiModal Rag


# -------------------------------------------------------------
# tools

def clean_chart(spec):
    if not spec or not spec.get("chart_type"):
        return None
    labels = spec['labels']
    values = spec['values']
    if len(labels) == 0 or len(values) == 0:
        return None
    return {
        "chart_type": spec["chart_type"],
        "title": spec.get("title", ""),
        "labels": spec["labels"],
        "values": spec["values"]
    }


# -------------------------------------------------------------
# text data loader
@app.post('/upload/text')
async def upload(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        print("Error")
    doc_id = str(uuid.uuid4())
    filepath = UPLOAD_DIR / f'{doc_id}_{file.filename}'
    with open(filepath, 'wb') as f:
        f.write(await file.read())

    try:
        loader = UnstructuredPDFLoader(filepath)
        print('Uploading documents...')
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)
    except Exception as e:
        print(e)
        return {"error": str(e)}

    for i, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["filename"] = file.filename
        chunk.metadata["chunk_index"] = i

    text_collection.add_documents(chunks)
    return f"Text uploaded"

# image data loader


image_collection = client.get_or_create_collection(
    name="images"
)
# ---------------------------temp fix script-------------------


# -------------------------------------------------------------

# Load CLIP once, not for every uploaded image.

clip_model, clip_preprocessor = clip.load(
    "ViT-B/32"
)

clip_model.eval()


@app.post('/upload/image')
async def upload_image(file: UploadFile):

    # 1. Validate the file extension.
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    filename = Path(file.filename or "").name

    if Path(filename).suffix.lower() not in allowed_extensions:
        raise ValueError("Unsupported image format")

    # 2. Read the uploaded image.
    image_id = str(uuid.uuid4())

    image_path = IMAGE_UPLOAD_DIR / f"{image_id}_{filename}"

    contents = await file.read()

    if not contents:
        raise ValueError("Uploaded image is empty")

    with open(image_path, "wb") as f:
        f.write(contents)

    try:
        with Image.open(image_path) as img:
            img.verify()

        # Reopen because verify() checks the file but does not
        # leave the image ready for normal processing.
        with Image.open(image_path) as img:
            image = img.convert("RGB")

    except (UnidentifiedImageError, OSError):
        image_path.unlink(missing_ok=True)
        raise ValueError("Invalid or corrupted image")

    # 3. Preprocess and encode the image.
    image_tensor = clip_preprocessor(image).unsqueeze(0)

    with torch.inference_mode():
        embedding = clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True
        )

    # 4. Convert the embedding into a list for ChromaDB.
    vector = embedding.numpy()[0].tolist()

    # 5. Store the vector and its metadata.
    image_collection.add(
        ids=[image_id],
        embeddings=[vector],
        metadatas=[{
            "filename": filename,
            "image_path": str(image_path),
            "modality": "image"
        }]
    )

    return {
        "image_id": image_id,
        "filename": filename,
        "status": "indexed"
    }


def image_data(path):
    with open(path, 'rb') as image_path:
        return base64.b64encode((image_path).read()).decode("utf-8")


@app.post('/upload/tables')
async def upload_table(file: UploadFile):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    filepath = UPLOAD_DIR / f'{str(uuid.uuid4())}_{file.filename}'
    with open(filepath, 'wb') as f:
        f.write(content)
    documents = []

    for index, row in df.iterrows():
        row_text = ', '.join(
            f"column: {column}: value: {value}" for column, value in row.items())

        document = Document(page_content=row_text, metadata={
                            'source': file.filename, 'row': int(index)})
        documents.append(document)

    table_collection.add_documents(documents)
    return {
        "message": "CSV uploaded successfully",
        "filename": file.filename,
        "rows_processed": len(documents)
    }

# -------------------------------------------------------------

# model


def image_retrival(query):
    inputs = clip.tokenize([query], truncate=True)
    with torch.no_grad():
        query_embedding = clip_model.encode_text(inputs)
    query_embedding = f.normalize(query_embedding, p=2, dim=-1)
    results = image_collection.query(
        query_embeddings=query_embedding.cpu().numpy().tolist(), n_results=min(3, image_collection.count()))
    print("Image collection count:", image_collection.count())
    print("Image collection data:", image_collection.get())
    image_paths = []
    for metadata in results["metadatas"][0]:
        image_paths.append(metadata["image_path"])
    return image_paths

#    return results


@app.post('/chat', response_model=FireSafetyResponse)
async def chat(request: ChatRequest):
    # image context retrieval
    image_context_paths = image_retrival(request.question)
    image_data_list = []
    for image in image_context_paths:
        image_data_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data(image)}"
            }
        })

    # text context retrieval
    text_docs = text_collection.similarity_search(query=request.question, k=5)
    text_context = "\n\n".join(
        f"Source: {doc.metadata.get('filename', 'Unknown')}\n"
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in text_docs
    )

    # table context retrieval
    table_docs = table_collection.similarity_search(
        query=request.question, k=5)
    experimental_data = "\n\n".join(
        f"Source: {doc.metadata.get('filename', 'Unknown')}\n"
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in table_docs
    )

    SYSTEM_PROMPT = """
You are a scientific research assistant specializing in combustion,
flame behavior in microgravity, and fire safety for future human
space exploration.

Your purpose is to help researchers understand NASA fire experiments
and interpret their findings for fire safety on spacecraft, the Moon,
and Mars.

You will receive:
1. A user's question.
2. Retrieved text from NASA research documents.
3. Retrieved experimental data, when available.
4. Metadata about relevant images, when available.

Use the provided information to answer the user's question accurately
and scientifically.

RULES:

1. EVIDENCE AND ACCURACY
- Base your answers on the retrieved evidence.
- Do not invent experimental results, measurements, citations,
  experiment identifiers, or scientific findings.
- If the available evidence is insufficient, explicitly state that.
- Clearly distinguish established findings from interpretations,
  hypotheses, and potential implications.

2. SCIENTIFIC INTERPRETATION
- Explain scientific concepts clearly and accurately.
- When relevant, discuss flame shape, flame spread, combustion,
  oxygen availability, heat transfer, and the effects of gravity.
- Distinguish observations made in microgravity from predictions
  about lunar or Martian gravity.
- Do not assume that results obtained in microgravity apply directly
  to the Moon or Mars.

3. EXPERIMENTAL DATA
- Include experimental measurements only when supported by the
  retrieved evidence.
- Preserve the original values and units.
- Do not invent missing values or units.
- Identify the relevant experiment when that information is available.
- If no relevant measurements are available, return an empty list
  rather than fabricating data.

4. SOURCES AND CITATIONS
- Identify the source document for important findings whenever
  the information is available.
- Include page numbers and experiment identifiers when provided.
- Do not invent source names, page numbers, or citations.
- Make it clear which evidence supports each finding.

5. IMAGES
- Use image metadata only to identify and reference retrieved images.
- Do not claim to have visually examined an image if only its filename,
  path, or metadata was provided.
- Do not infer scientific findings from image metadata alone.
- If relevant image information is unavailable, return an empty list
  for related images.

6. SPACE FIRE SAFETY
- When relevant, explain how the findings might inform fire prevention,
  fire detection, material selection, or emergency response during
  future space missions.
- Clearly label potential applications as implications rather than
  proven experimental results.
- Do not make unsupported claims about spacecraft, lunar habitats,
  or Martian habitats.

7. RESPONSE FORMAT
- Return your answer using the provided FireSafetyResponse schema.
- Populate each field with the appropriate information.
- Use empty lists for list fields when no relevant evidence is available.
- Keep answers clear, focused, and scientifically useful.
- Include relevant limitations and uncertainties.
- Do not add fields that are not part of the required schema.
- Return your entire response as valid JSON.
- Include every required field in the FireSafetyResponse schema.
- Do not omit the answer field, even if some other fields are empty.

For all page fields:
- Return page numbers as integers only.
- If the page number is unknown, return null.
- Never return "Unknown", "N/A", or other text in page fields.

8. CHART GENERATION
- Always look for opportunities to create a meaningful chart using the retrieved research data.
- Do not restrict chart generation to flame spread rates or any specific measurement.
- Use any available numerical data, including temperature, flame size, oxygen concentration, burning duration, ignition time, and experimental comparisons.
- If the requested metric is unavailable, look for other numerical measurements in the retrieved documents.
- If numerical data is unavailable but meaningful qualitative comparisons exist, create a clearly labelled qualitative comparison chart where appropriate.
- Never invent experimental measurements or present AI-generated estimates as actual experimental results.
- Use bar charts for comparisons, line charts for trends over time, and pie charts for proportions.
- Include a clear title, labels, and units where applicable.
- Return chart=None only when no meaningful visualization can be created from the available evidence.
- Always explain the chart and its significance in the written response.

Your priority is scientific accuracy, evidence-based reasoning,
and transparent communication of uncertainty.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", """
    Question:
    {question}

    Retrieved NASA text:
    {text_context}

    Retrieved experimental data:
    {experimental_data}

    Retrieved image metadata:
    {image_context}
    """)
    ])

    structured_llm = llm.with_structured_output(FireSafetyResponse)

    content = [
        {
            "type": "text",
            "text": f"""
        Question: {request.question}

        Retrieved NASA text:
        {text_context}

        Retrieved experimental data:
        {experimental_data}

        Retrieved image information:
        {image_context_paths}

        Analyze the actual images alongside the retrieved research.
        """
        }
    ]
    content.extend(image_data_list)

    response = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=content)
    ])
    chart_path = None
    data = response.model_dump()
    data["charts"] = clean_chart(data.get("charts"))

    for item in data['related_images']:
        for image_path in image_context_paths:
            filename = Path(image_path).name
            if item["image_id"] == filename:

                item["image_url"] = (
                    f"/uploaded_images/{filename}"
                )

                break

    return data
