from flask import Flask, request, jsonify 
from flask_cors import CORS
import torch
from torch import nn
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np 
import faiss
# ==================== Загрузка и инициализация моделей ====================
#Загрузка модели для распознавания видео 
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True)
# Загрузка модели для преобразования текста в эмбеддинги
model_path = "./winstrike_rutube_tags_model"
model_text = SentenceTransformer(model_path, )
# Размер вектора эмбеддинга (соответствует используемой текстовой модели)
dim = 768 

# Загрузка предварительно вычисленных эмбеддингов тегов
with open('./tags_embeddings.npy', "rb") as f:
    np_tags_arr = np.load(f)

# Загрузка списка всех возможных тегов
with open('./organized_tags.txt', "r", encoding="utf-8") as f:
    all_tags_list = f.read().split('\n')
# ==================== Импорт необходимых библиотек ====================
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import urllib.request
import json
import os
from werkzeug.utils import secure_filename
import warnings
from PIL import Image
import random
import io
import base64
import cv2

## Отключение предупреждений для чистоты вывода
warnings.filterwarnings("ignore") 

# Инициализация приложения Flask и разрешение CORS
app = Flask(__name__)
CORS(app)

# Настройка папки для загрузки видео
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== Конфигурация модели ====================

# Установка устройства: CPU или GPU (при наличии)
device = "cpu"
model = model.eval()
model = model.to(device)

# Загрузка названий классов из датасета Kinetics
json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try:
    urllib.request.urlretrieve(json_url, json_filename)
except Exception as e:
    print(f"Ошибка при загрузке файла классов: {e}")

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Создание отображения id в названия классов
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

# Параметры преобразований видео
side_size = 256 # Размер короткой стороны для масштабирования
mean = [0.45, 0.45, 0.45] # Среднее для нормализации
std = [0.225, 0.225, 0.225] # Стандартное отклонение для нормализации
crop_size = 256 # Размер для центрирования и обрезки
num_frames = 32 # Количество кадров для выборки
sampling_rate = 2 # Частота выборки кадров
frames_per_second = 30 # Частота кадров в видео
slowfast_alpha = 4 # Параметр alpha для модели SlowFast

#Функция получения тегов по описанию
"""
    Предсказывает теги по заголовку и описанию видео с помощью текстовых эмбеддингов.

    Аргументы:
        title (str): Заголовок видео.
        description (str): Описание видео.

    Возвращает:
        list: Список предсказанных тегов.
"""
def get_predict_from_title_and_description(title : str, description : str) -> list:
    """ Получение тегов к видео по наименованию и описанию """
    global model_text, dim, np_tags_arr
    
    embedding = model_text.encode(title + " " + description, convert_to_tensor=True).cpu().numpy()
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(np_tags_arr)


    topn = 3
    scores, predictions = index.search(np.array([list(embedding)], dtype=np.float64), topn)


    return scores[0], [all_tags_list[idx] for idx in predictions[0]]

#Функция получения тегов следующего уровня по соотношению тегов полученных с видео
"""
    Комбинирует теги из содержимого видео и текста для получения финальных тегов.

    Аргументы:
        title (str): Заголовок видео.
        description (str): Описание видео.
        lvl_1_tags (list[str]): Теги, предсказанные по содержимому видео.

    Возвращает:
        list[str]: Финальный список тегов.
"""
def get_tags(title : str, description : str, lvl_1_tags : list[str]) -> list[str]:
    scores, tags_from_text = get_predict_from_title_and_description(title, description)

    result_tags = []

    # отфильтрорываем теги, на которых модель сомневается
    tags_from_text = [tag for (idx, tag) in enumerate(tags_from_text) if int(scores[idx]) > 250]

    if not len(tags_from_text):
            return lvl_1_tags

    for tag_from_text in tags_from_text:
        for tag_from_video in lvl_1_tags:
            if (tag_from_video in tag_from_text):
                result_tags.append(tag_from_text)
                break
    
    if not len(result_tags):
        return tags_from_text
    
    return result_tags

class PackPathway(torch.nn.Module):
    
    #Подготавливает входные данные для модели SlowFast, создавая медленный и быстрый пути.
    
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Выполнение временной выборки для медленного канала.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        return [slow_pathway, fast_pathway]
    
# Определение конвейера преобразований для кадров видео
transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),# Равномерная выборка кадров
            Lambda(lambda x: x / 255.0),# Нормализация значений пикселей
            NormalizeVideo(mean, std),# Нормализация видео
            ShortSideScale(size=side_size),# Изменение размера видео
            CenterCropVideo(crop_size),# Центрированная обрезка
            PackPathway() # Подготовка входных данных для модели SlowFast
        ]
    ),
)

# Расчет длительности клипа для обработки
clip_duration = (num_frames * sampling_rate) / frames_per_second

# Определение категорий и связанных с ними действий для классификации
categories = {
    # Транспорт
    'Транспорт': ['changing wheel', 'driving car', 'driving tractor', 'riding scooter', 'golf driving', 'riding a bike', 'pushing car', 'motorcycling', 'unloading truck', 'changing oil', 'checking tires'],
    # Книги и литература
    'Книги и литература': ['sharpening pencil', 'reading newspaper', 'reading book'],
    # Бизнес и финансы
    'Бизнес и финансы': ['counting money', 'auctioning'],
    # Карьера
    'Карьера': [],
    # Образование
    'Образование': ['writing'],
    # События и достопримечательности
    'События и достопримечательности': ['clapping', 'drinking beer', 'tasting beer', 'balloon blowing', 'bartending', 'bowling', 'playing poker', 'celebrating', 'drinking shots', 'applauding', 'smoking hookah'],
    # Семья и отношения
    'Семья и отношения': ['flying kite', 'blowing out candles', 'braiding hair', 'hopscotch', 'ironing', 'dining', 'making snowman', 'kissing', 'crawling baby', 'carrying baby', 'baby waking up'],
    # Изобразительное искусство
    'Изобразительное искусство': ['somersaulting', 'sharpening pencil', 'spray painting', 'brush painting'],
    # Еда и напитки
    'Еда и напитки': ['sharpening knives', 'eating ice cream', 'eating carrots', 'cooking on campfire', 'breading or breadcrumbing', 'drinking', 'cooking egg', 'eating spaghetti', 'making pizza', 'peeling potatoes', 'flipping pancake', 'drinking beer', 'tasting beer', 'making sushi', 'making a sandwich', 'cutting pineapple', 'bartending', 'frying vegetables', 'eating watermelon', 'eating doughnuts', 'drinking shots', 'peeling apples', 'eating burger', 'eating hotdog', 'scrambling eggs', 'cooking chicken', 'cooking sausages', 'tossing salad', 'barbequing', 'cutting watermelon', 'making tea', 'tasting food', 'making a cake', 'eating chips', 'baking cookies', 'grinding meat', 'eating cake'],
    # Здоровый образ жизни
    'Здоровый образ жизни': ['bench pressing', 'deadlifting', 'eating carrots', 'playing badminton', 'jogging', 'running on treadmill', 'push up', 'pull ups', 'skipping rope'],
    # Хобби и интересы
    'Хобби и интересы': ['sharpening knives', 'cutting nails', 'playing flute', 'playing saxophone', 'abseiling', 'rock climbing', 'golf putting', 'tai chi', 'skydiving', 'ice fishing', 'playing chess', 'water sliding', 'roller skating', 'snowboarding', 'catching fish', 'playing paintball'],
    # Дом и сад
    'Дом и сад': ['bee keeping', 'peeling potatoes', 'driving tractor', 'feeding goats', 'sanding floor', 'trimming trees', 'mowing lawn', 'decorating the christmas tree', 'shoveling snow', 'planting trees', 'watering plants', 'digging', 'arranging flowers', 'chopping wood'],
    # Медицина
    'Медицина': ['sticking tongue out', 'bandaging', 'pushing wheelchair', 'gargling', 'massaging person\'s head', 'brushing teeth'],
    # Фильмы и анимация
    'Фильмы и анимация': ['smoking', 'kissing', 'crying', 'laughing'],
    # Музыка и аудио
    'Музыка и аудио': ['playing flute', 'playing saxophone', 'playing cymbals', 'playing organ', 'playing bagpipes', 'playing clarinet', 'playing piano', 'dancing charleston', 'playing accordion', 'playing trumpet', 'recording music', 'drumming fingers', 'playing ukulele', 'playing violin', 'playing drums', 'playing harmonica', 'playing xylophone', 'playing cello', 'singing', 'tapping guitar', 'strumming guitar', 'playing guitar', 'playing harp', 'beatboxing'],
    # Новости и политика
    'Новости и политика': ['extinguishing fire', 'presenting weather forecast', 'news anchoring', 'reading newspaper'],
    # Личные финансы
    'Личные финансы': [],
    # Животные
    'Животные': ['milking cow', 'holding snake', 'bee keeping', 'feeding goats', 'petting cat', 'training dog', 'grooming dog', 'riding elephant', 'grooming horse', 'petting animal (not cat)', 'riding mule', 'walking the dog', 'shearing sheep', 'feeding birds'],
    # Недвижимость
    'Недвижимость': [],
    # Религия и духовность
    'Религия и духовность': [],
    # Наука
    'Наука': [],
    # Покупки
    'Покупки': [],
    # Спорт
    'Спорт': ['bench pressing', 'deadlifting', 'throwing discus', 'playing badminton', 'capoeira', 'hurling (sport)', 'wrestling', 'gymnastics tumbling', 'ice skating', 'jogging', 'bobsledding', 'golf putting', 'dribbling basketball', 'hockey stop', 'situp', 'playing basketball', 'high kick', 'playing chess', 'snowboarding', 'push up', 'throwing ball', 'archery', 'snatch weight lifting', 'playing ice hockey', 'swimming butterfly stroke', 'pull ups', 'punching bag', 'skiing crosscountry', 'punching person (boxing)', 'playing tennis', 'dunking basketball', 'arm wrestling', 'kicking soccer ball', 'shooting basketball', 'side kick', 'playing volleyball'],
    # Стиль и красота
    'Стиль и красота': ['cutting nails', 'making jewelry', 'trimming or shaving beard', 'brushing hair', 'waxing legs', 'curling hair', 'shaving legs', 'getting a haircut', 'braiding hair', 'dying hair', 'getting a tattoo', 'fixing hair', 'filling eyebrows', 'waxing back', 'waxing chest', 'doing nails', 'shaving head', 'waxing eyebrows', 'applying cream'],
    # Информационные технологии
    'Информационные технологии': ['texting', 'using computer', 'using remote controller (not gaming)', 'robot dancing'],
    # Телевидение
    'Телевидение': ['presenting weather forecast', 'news anchoring', 'giving or receiving award', 'answering questions'],
    # Путешествия
    'Путешествия': ['marching', 'cooking on campfire', 'abseiling', 'rock climbing', 'canoeing or kayaking', 'skydiving', 'ice fishing', 'riding mountain bike', 'riding camel', 'snorkeling', 'biking through snow', 'catching fish', 'riding elephant', 'riding a bike'],
    # Игры
    'Игры': ['assembling computer', 'using computer', 'playing keyboard', 'playing controller', 'playing cards'],
}

# Инициализация хранилища видео 
videos = []

# ==================== Функция для извлечения кадра ====================
"""
    Извлекает случайный кадр из центральной части видео.

    Аргументы:
        video_path (str): Путь к видеофайлу.

    Возвращает:
        str: Base64-кодированная строка изображения.
"""
def extract_random_frame(video_path):
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Получаем общее количество кадров
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    # Определяем диапазон середины видео (25% до 75%)
    start_frame = int(total_frames * 0.25)
    end_frame = int(total_frames * 0.75)

    # Выбираем случайный кадр внутри диапазона
    random_frame_number = random.randint(start_frame, end_frame)

    # Устанавливаем позицию кадра
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)

    # Читаем кадр
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    # Конвертируем кадр в RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Преобразуем в изображение PIL
    image = Image.fromarray(frame)

    # Конвертируем изображение в байты
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str
# ==================== Функция обработки видео ====================
"""
    Обрабатывает видео и предсказывает теги первого уровня на основе содержимого видео.

    Аргументы:
        video_path (str): Путь к видеофайлу.

    Возвращает:
        tuple: (список предсказанных тегов, сообщение об ошибке или None)
"""
def process_video(video_path):
    """
    Обрабатывает видео и возвращает список тегов tag_level1.
    """
    try:
        # Инициализация EncodedVideo и загрузка видео
        video = EncodedVideo.from_path(video_path)

        # Общая продолжительность видео
        total_duration = video.duration

        # Проверяем, что видео удалось загрузить
        if total_duration is None:
            return None, 'Не удалось загрузить видео'

        # Список для накопления предсказаний
        all_preds = []

        start_sec = 0

        # Обработка видео по клипам
        while start_sec < total_duration:
            end_sec = start_sec + clip_duration
            if end_sec > total_duration:
                end_sec = total_duration

            # Загрузка клипа
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

            # Проверяем, что клип содержит данные
            if video_data is None or video_data["video"] is None:
                break

            # Применение преобразований
            video_data = transform(video_data)

            # Подготовка входных данных для модели
            inputs = video_data["video"]
            inputs = [i.to(device)[None, ...] for i in inputs]

            # Получение предсказаний от модели
            with torch.no_grad():
                preds = model(inputs)

            # Добавление предсказаний в список
            all_preds.append(preds.cpu())

            # Переход к следующему клипу
            start_sec += clip_duration

        # Проверяем, были ли получены предсказания
        if not all_preds:
            return None, 'Не удалось обработать видео'

        # Агрегация предсказаний по всему видео
        final_preds = torch.cat(all_preds).mean(dim=0, keepdim=True)

        # Применение Softmax для получения вероятностей
        post_act = torch.nn.Softmax(dim=1)
        final_preds = post_act(final_preds)

        # Получение топ-5 предсказанных классов
        pred_classes = final_preds.topk(k=5).indices[0]

        # Отображение предсказанных классов на названия
        pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]

        # Преобразование названий действий в нижний регистр для корректного сравнения
        pred_class_names_lower = [name.lower() for name in pred_class_names]

        # Инициализация списка категорий
        tag_level1 = []

        # Проверка предсказаний на соответствие категориям
        for category, actions in categories.items():
            # Преобразуем действия в нижний регистр для сравнения
            actions_lower = [action.lower() for action in actions]
            if any(pred in actions_lower for pred in pred_class_names_lower):
                tag_level1.append(category)

        # Если ни одна категория не совпала, добавляем 'другое'
        if not tag_level1:
            tag_level1.append('другое')
        
        return tag_level1, None  # Возвращаем список категорий и None (ошибки нет)

    except Exception as e:
        return None, str(e)  # Возвращаем None и сообщение об ошибке

# ==================== Маршруты Flask ====================

@app.route('/upload', methods=['POST'])
def upload_video():
    # Получаем загруженное видео
    if 'video' not in request.files:
        return jsonify({'error': 'Видео не предоставлено'}), 400

    video_file = request.files['video']
    title = request.form.get('title', '')
    description = request.form.get('description', '')

    if video_file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    # Сохраняем видео во временную папку
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    try:
        # Обработка видео и получение тегов
        tag_level1, error = process_video(video_path)
        if error:
            raise Exception(error)
        results = get_tags(title, description, tag_level1)
        # Извлекаем случайный кадр из середины видео
        frame_image_base64 = extract_random_frame(video_path)

        # Формируем данные для ответа и сохранения
        response_data = {
            'title': title,
            'description': description,
            'categories': results,
            'status': 'успешно',
            'frame_image': frame_image_base64  # Включаем изображение кадра
        }

        # Сохраняем информацию о видео в список
        video_info = {
            'video_id': len(videos) + 1,
            'title': title,
            'description': description,
            'categories': results,
            'status': 'успешно',
            'frame_image': frame_image_base64  # Включаем изображение кадра
        }
        videos.append(video_info)

        # Возвращаем результаты
        return jsonify(response_data), 200

    except Exception as e:
        # В случае ошибки сохраняем информацию о неудачной обработке
        error_info = {
            'video_id': len(videos) + 1,
            'title': title,
            'description': description,
            'categories': [],
            'status': 'ошибка',
            'error_message': str(e),
            'frame_image': None  # Нет изображения в случае ошибки
        }
        videos.append(error_info)
        return jsonify({'error': str(e)}), 500

    finally:
        # Удаляем загруженный файл после обработки
        if os.path.exists(video_path):
            os.remove(video_path)

# ==================== Маршрут для получения списка видео ====================

@app.route('/videos', methods=['GET'])
def get_videos():
    return jsonify({'videos': videos}), 200

if __name__ == '__main__':
    app.run(debug=True)
