import os

TRAIN_DATA_FOLDER = "../datasets/train-articles/"

chars = set()
article_file_names = [file_name for file_name in os.listdir(TRAIN_DATA_FOLDER)
                      if file_name.endswith(".txt")]
for file_name in article_file_names:
    with open(TRAIN_DATA_FOLDER + '/' + file_name, encoding='utf-8') as f:
        chars.update(f.read())

chars.remove('\n')
print(sorted(list(chars)))
