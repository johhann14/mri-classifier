import pandas as pd
import os


def create_inital_df(base_dir):
    data = []

    for split in ['Training', 'Testing']:
        split_dir = os.path.join(base_dir, split)

        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if os.path.isdir(label_dir):
                for img in os.listdir(label_dir):
                    img_path = os.path.abspath(os.path.join(label_dir, img))
                    data.append({
                        'path': img_path,
                        'label': label,
                        'split': split.lower()
                    })

    return pd.DataFrame(data)
