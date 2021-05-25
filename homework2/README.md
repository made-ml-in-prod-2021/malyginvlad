### Запуск проекта
#### Нулевой этап:
```shell script
$ git clone https://github.com/made-ml-in-prod-2021/malyginvlad.git
$ cd malyginvlad/homework2
```
#### Дальше необходимо установить текущий проект:
```shell script
$ pip install -e .
```
#### Теперь можно запускать докер:
```shell script
$ docker build -t malyginvlad/made_ml_prod_homework2:latest . 
$ docker run -p 8000:8000 malyginvlad/made_ml_prod_homework2:latest
```
----------
#### Или скачать образ с докер-хаба:
```shell script
$ docker pull malyginvlad/made_ml_prod_homework2:latest
$ docker run -p 8000:8000 malyginvlad/made_ml_prod_homework2:latest
```
----------
#### Запрос к системе:
```shell script
$ python make_request.py
```
----------
#### Запуск тестов:
```shell script
$ pytest test_pipeline.py
```
----------
#### Оптимизация докер образа:
Изначально строил образ на `python:3.6` - получилось 1.29GB. Попробовал вместо этого `python:3.6-slim` - стало 524MB.
