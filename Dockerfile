FROM python:3.10.14

WORKDIR /app

COPY require.txt ./require.txt

RUN pip3 install -r require.txt 

COPY app.py ./app.py

COPY kaggle ./kaggle

COPY Aimodel.h5 ./Aimodel.h5

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]