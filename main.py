from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx
from contextlib import asynccontextmanager
from sqlmodel import Session, create_engine, SQLModel, select, text, Field
from sqlalchemy import inspect
from typing import Annotated, List, Any, Optional
from fastapi import Depends
from pydantic import BaseModel
from pgvector.sqlalchemy import Vector
import pymupdf
from google import genai
from dotenv import load_dotenv
import time
import os
from google.genai.errors import ClientError
import instructor

load_dotenv()

#App initialization
llm_client = instructor.from_provider("google/models/gemini-2.5-flash-lite-preview-06-17")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = FastAPI()
postgres_url = "postgresql://user:password@db:5432/my_database"
engine = create_engine(postgres_url)

#Models
class PDFPage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    page_number: int
    page_text: str
    embedding: Any = Field(sa_type=Vector(3072))

def get_session():
    with Session(engine) as session:
        yield session

#Database initialization
def create_db_and_tables():
    #inspector = inspect(engine)
    #existing_tables = inspector.get_table_names()
    #is_first_time = not existing_tables
    with Session(engine) as session:
        session.exec(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.commit()
    SQLModel.metadata.create_all(engine)
    #if is_first_time:
    #    print("Creating tables")

#App lifetime manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI(lifespan=lifespan)

#Utils functions, LLM calls, etc.
def generate_embeddings_and_save_to_pgvector(text: List[str], session: Session):
    '''
    Generates embeddings for a list of text chunks and saves them to the database.
    '''
    with session.begin():
        page_number = 0
        for text in text:
            if len(text) < 10 or text.isspace():
                continue
            try:
                embedding = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                )
                page_number += 1
                page = PDFPage(page_number=page_number, page_text=text, embedding=embedding.embeddings[0].values)
                session.add(page) 
                print('adding embedding')
            except ClientError:
                try:
                    print('ClientError, waiting 120 seconds')
                    time.sleep(120)
                    embedding = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=text,
                    )
                    page_number += 1
                    page = PDFPage(page_number=page_number, page_text=text, embedding=embedding.embeddings[0].values)
                    session.add(page) 
                    print('adding embedding')
                except Exception as e:
                    print('Unexpected error: ', e)
                    continue
            except Exception as e:
                print('Unexpected error: ', e)
                continue
        print('Trying to commit')
        session.commit()
        print('COMMITED')

async def generate_embeddings(url: str, session: Session):
    text_list = []
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF from URL.")
        pdf_content = response.content
        doc = pymupdf.Document(stream=pdf_content)
        for page in doc:
            page_text = page.get_text()
            text_list.extend([page_text[i:i+1000] for i in range(0, len(page_text), 1000)])

        generate_embeddings_and_save_to_pgvector(text_list, session)

def generate_question_embedding(question: str):
    embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
    )
    return embedding.embeddings[0].values

#Routes
@app.get("/")
async def root():
    return {"message": "Hello World2"}

class PDFUpload(BaseModel):
    url: str

@app.post("/upload_pdf")
async def upload_pdf(pdf_file: PDFUpload, session: SessionDep, background_tasks: BackgroundTasks):
    url = pdf_file.url
    if not url.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")
    #download pdf from url
    background_tasks.add_task(generate_embeddings, url, session)
    
    return {'message': 'Saving pdf to pgvector...'}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatQuestion(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None

@app.post("/ask_pdf")
async def ask_pdf(question: ChatQuestion, session: SessionDep):
    #Generating question embedding and searching for similar pages
    question_embedding = generate_question_embedding(question.question)
    pdf_pages = session.exec(select(PDFPage).order_by(PDFPage.embedding.l2_distance(question_embedding)).limit(5)).all()

    context = 'Here are snippets from a pdf file related to the question. Answer the question based on the relevant context: \n\n'
    for pdf_page in pdf_pages:
        context += f'{pdf_page.page_text}\n'

    #Generating chat history
    history = question.history
    if history is None:
        history = []
    #history.append({"role": "user", "content": question.question})
    history.insert(0, {"role": "system", "content": "You are a helpful assistant that answers questions based on snippets from a pdf file. Always try to reference the given context to answer the user. If you don't know the answer, say so."})
    history.append({"role": "user", "content": f"Question: {question}\nContext: {context}"})

    response = llm_client.chat.completions.create(
        messages=history,
        response_model=str
    )

    return {'message': response, 'question': question.question, 'context': context}