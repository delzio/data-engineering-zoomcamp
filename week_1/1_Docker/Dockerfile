FROM python:3.10

RUN pip install pandas

# can specify directory in container to add files
WORKDIR /app
COPY pipeline.py pipeline.py

ENTRYPOINT [ "python", "pipeline.py" ]