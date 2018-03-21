import os
import pandas as pd
import cv2


def img_to_csv(path):
    img_list = []
    for className in os.listdir(path):
        subpath = os.path.join(path, className)
        for filename in os.listdir(subpath):
            img_path = os.path.join(subpath, filename)
            img = cv2.imread(img_path)
            if img is None: continue
            w, h, ch = img.shape
            img_list.append((filename, w, h, className, int(w/7), int(h/7), w-int(w/7), h-int(h/7)))
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    img_df = pd.DataFrame(img_list, columns=column_name)
    return img_df


if __name__ == '__main__':
    image_path = os.path.join(os.getcwd(), 'data/Marcel-Train')
    xml_df = img_to_csv(image_path)
    xml_df.to_csv('data/object_labels.csv', index=None)
    print('Successfully converted img data to csv.')