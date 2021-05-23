## Homework №2

~~~
cd online_inference
python -m venv .venv .venv\Scripts\activate.bat pip install -r requirements.txt
~~~

Run app:
~~~
python src/app.py
~~~
Run request script:
~~~
python src/make_request.py
~~~
Tests:
~~~
pytest tests
~~~

Docker build command:
~~~
docker build -t shubinsa/inference:v2 .
~~~
Docker run command:
~~~
docker run -p 8000:8000 shubinsa/inference:v2
~~~
Docker push command:
~~~
docker push shubinsa/inference:v2
~~~
Docker pull command:
~~~
docker pull shubinsa/inference:v2
~~~
