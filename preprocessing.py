import os
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split


class DataPreprocessor:
    '''Разделитель тренировочного датасета с двумя и более классами на пропорциональные выборки train, test и valid'''
    def __init__(self, val_size: float = 0.15, test_size: float = 0.15):
        self.val_size = val_size
        self.test_size = test_size

    def split_roboflow_dataset(self, annotations_csv: str, output_dir: str):
        if not os.path.exists(annotations_csv):
            raise "Bad path: Annotations csv not found"

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        df_annots = pd.read_csv(annotations_csv)

        for field in ['filename', 'class']:
            if field not in df_annots.columns:
                raise f"Bad content: '{field}' column in annotations csv not found"

        file_bestclass = df_annots.groupby('filename')['class'].agg(list).apply(
            lambda s: Counter(s).most_common()[0][0]
        )

        matching_table = pd.DataFrame(file_bestclass).reset_index()
        matching_table = matching_table.rename({'class': 'major_class'}, axis=1)

        # split train filenames from val and test
        files_train, files_valtest, _, class_valtest = train_test_split(
            matching_table, matching_table['major_class'],
            test_size=(self.val_size + self.test_size),
            random_state=222,
            stratify=matching_table['major_class']
        )

        # split test filenames from val
        files_val, files_test, y_val, y_test = train_test_split(
            files_valtest, class_valtest,
            test_size=self.test_size / (self.val_size + self.test_size),
            random_state=222,
            stratify=class_valtest
        )

        files_train['split'] = 'train'
        files_val['split'] = 'val'
        files_test['split'] = 'test'

        files_merged = pd.concat([files_train, files_val, files_test]).set_index('filename')

        df_annots['split'] = df_annots['filename'].apply(
            lambda f: files_merged.loc[f, 'split']
        )

        if (df_annots.groupby('filename')['split'].agg('nunique') != 1).sum() != 0:
            raise "Split not successful: objects of one filename have different split directories"

        df_annots[df_annots['split'] == 'train'].to_csv(os.path.join(output_dir, 'train_split.csv'))
        df_annots[df_annots['split'] == 'val'].to_csv(os.path.join(output_dir, 'val_split.csv'))
        df_annots[df_annots['split'] == 'test'].to_csv(os.path.join(output_dir,'test_split.csv'))










