FROM python:3.7
WORKDIR '/app'

RUN pip install numpy pandas seaborn sklearn imblearn matplotlib tabulate

CMD ["/app/mount/start.sh", "--host=0.0.0.0"]

#docker build -t python-yelpanalysis .
#docker run --mount type=bind,source="%cd%"\mount,target=/app/mount python-yelpanalysis