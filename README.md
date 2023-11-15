# Folder_sorter  
 Модуль для предобработки и создания конечного датасета 

## Инструкция
1. Необходимо поместить в корневую папку с файлом `main.py` датасет выгруженного с *Roboflow* в формате *\*Tensorflow.zip* без аугментации  

2. Необходимо выгрузить с *Roboflow* датасет с аугментацией без разбиения на  
выборки train/test/val в формате *\*YOLOv8.zip* в корневую директорию  
рядом с файлом `main.py`.  

3. Устанавливаем зависимости:

```
pip install -r requirements.txt
```

4. Запускаем программу:

```
python main.py
```

_Полученный датасет AIWDB_yolov8_sc.zip далее используем для работы с моделью YOLOv8_
