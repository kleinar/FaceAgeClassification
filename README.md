# Human age classification 

![Image alt](https://github.com/kleinar/FaceAgeClassification/raw/master/misc/group.jpg)

# Train age classification

1. Скачать датасет по лицам по ссылке (тут ссылку)
2. Закинуть папку с датасетом в корень проекта
3. Зайти в config файл и прописать путь к датасету.
   Например: path-to-dataset: 'UTKFace_Dataset/'
4. В конфиг файле прописать параметры которые вы хотите написать.
   * 4.1. Есть возможность выбрать разные предобученные нейросети: resnet18, resnet34 и так далее.
   * 4.2. Поменять размер входного изображения для классификатора возраста людей
   * 4.3. Производить обучение на гпу или цпу и так далее
6. Запустить train.py

# Inference

1. После обучения вы получите best_model.pt в файле, в котором вы прописали в конфиге.
2. выберите модель, которую вы использовали как предобученную
3. пропишите путь к весам вашей нейросети
4. выберите, на чем хотите проверить ваш результат. На отдельном изображении, в папке из изображении или видео
5.  пропишите размер входного изображения как во время обучения
6.  Напишите, где хотите сохранять результаты

parser.add_argument('--model', nargs='+', type=str, default='resnet18', help='pretrained model name')
    parser.add_argument('--weights', nargs='+', type=str, default='models/best_model.pt', help='model path')
    parser.add_argument('--source', type=str, default= '', help='file/dir/video')
    parser.add_argument('--imgsz', type=list, default=224, help='224 or 512')
    parser.add_argument('--save_path', nargs='+', type=str, default='output/', help='save results path')
    
Запуск кода - python infer.py --model resnet18 --weights path-to-model --source path-to-img-dir-video --imgsz 224 --save_path path-to-save-results
