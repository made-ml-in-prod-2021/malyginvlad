## Homework 1

Данные для проекта: [датасет](https://www.kaggle.com/ronitf/heart-disease-uci)

Данный датасет необходимо положить в папку ***ml_project/data/raw***

Инструкция для Windows:
* Клонируем репозиторий проекта: ```git clone https://github.com/made-ml-in-prod-2021/malyginvlad.git```

* Переходим в папку проекта: ```cd malyginvlad```

* Меняем ветку, если необходимо: ```git checkout homework1```

* Создадим виртуальное окружение: ```python -m venv .venv```
  
* Активируем виртуальное окружение: ```.venv\Scripts\activate```
  
* Установим все необходимые зависимости: ```pip install .```

* Запускаем обучение: ```python ml_project/train_pipeline.py ml_project/configs/train_config.yaml```

* Запускаем тесты: ```python -m pytest ml_project/tests/```


## Homework 2

Инструкция внутри проекта
