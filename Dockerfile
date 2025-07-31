FROM python:3

WORKDIR .

COPY ./requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

CMD ["fastapi", "run", "--port", "8000"]