FROM python:3.10

RUN apt-get install wget
RUN pip install pandas sqlalchemy psycopg2

# can specify directory in container to add files
WORKDIR /app
COPY ingest_data.py ingest_data.py

ENTRYPOINT [ "python", "ingest_data.py" ]