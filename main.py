import csv
import os
import shutil
import argparse
from preprocessing import DataPreprocessor


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--annotations', default='./_annotations.csv', type=str)
parser.add_argument('-o', '--outdir', default='.', type=str)

args = parser.parse_args()


preprocessor = DataPreprocessor()

preprocessor.split_roboflow_dataset(
    annotations_csv=args.annotations,
    output_dir=args.outdir
)


# функция для переноса картинок и аннотаций в соответствующие выборке папки
def copy_and_remove(csv_file, source_folder, dest_image_folder, dest_label_folder):
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row['filename']
            image_path = os.path.join(source_folder, 'images', filename)
            label_path = os.path.join(source_folder, 'labels', filename[:-4] + '.txt')

            dest_image_path = os.path.join(dest_image_folder, filename)
            dest_label_path = os.path.join(dest_label_folder, filename[:-4] + '.txt')

            # Проверка, был ли файл уже скопирован
            if not os.path.exists(dest_image_path):
                # Копирование изображения
                shutil.copy(image_path, dest_image_path)
                os.remove(image_path)  # Удаление из исходной папки

            if not os.path.exists(dest_label_path):
                # Копирование файла меток
                shutil.copy(label_path, dest_label_path)
                os.remove(label_path)  # Удаление из исходной папки


if __name__ == "__main__":
    # test
    csv_file_test = 'test_split.csv'
    source_folder_test = '.'
    dest_image_folder_test = 'AIWDB_yolov8_sc/test/images'
    dest_label_folder_test = 'AIWDB_yolov8_sc/test/labels'

    # train
    csv_file_train = 'train_split.csv'
    source_folder_train = '.'
    dest_image_folder_train = 'AIWDB_yolov8_sc/train/images'
    dest_label_folder_train = 'AIWDB_yolov8_sc/train/labels'

    # valid
    csv_file_valid = 'val_split.csv'
    source_folder_valid = '.'
    dest_image_folder_valid = 'AIWDB_yolov8_sc/valid/images'
    dest_label_folder_valid = 'AIWDB_yolov8_sc/valid/labels'

    # Проверка и создание папок назначения, если они не существуют
    os.makedirs(dest_image_folder_test, exist_ok=True)
    os.makedirs(dest_label_folder_test, exist_ok=True)

    os.makedirs(dest_image_folder_train, exist_ok=True)
    os.makedirs(dest_label_folder_train, exist_ok=True)

    os.makedirs(dest_image_folder_valid, exist_ok=True)
    os.makedirs(dest_label_folder_valid, exist_ok=True)

    # Вызов функции для копирования файлов и удаления из исходной папки
    copy_and_remove(csv_file_test, source_folder_test, dest_image_folder_test, dest_label_folder_test)
    copy_and_remove(csv_file_train, source_folder_train, dest_image_folder_train, dest_label_folder_train)
    copy_and_remove(csv_file_valid, source_folder_valid, dest_image_folder_valid, dest_label_folder_valid)
