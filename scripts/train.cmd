cd C:\Users\nuaax\PycharmProjects\stance_detection_allan

rd /s /q C:\Users\nuaax\Dropbox\data61\project\stance_classification\dataset\semeval\models

python stance_detection\run.py train experiments/config_crossnet.json -s C:/Users/nuaax/Dropbox/data61/project/stance_classification/dataset/semeval/models