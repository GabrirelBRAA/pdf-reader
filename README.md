# How to Build
- add a .env file similar to `.env.example`. Both API keys can be the same one, they just have different names because GOOGLE_API_KEY will be used for LLM calls and GEMINI_API_KEY for embedding calls.
- run `docker compose up -d --build`

# How to interact
- At localhost:8000/docs you will have the swagger documentation.
- `/upload_pdf` takes an url and tries to download the pdf and create embeddings using the Google Embeddings Model.
- `/ask_pdf` embeds your query and does a vector search in the pgvector DB to find the text that most closely matches your query.
    It also takes in an optional history parameter which gives context to the LLM of the chat history if you want it.