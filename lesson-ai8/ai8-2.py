import torch  # Библиотека для машинного обучения и нейронных сетей
import whisper  # Модель для распознавания речи от OpenAI
import clip  # Модель для анализа изображений от OpenAI (связывает изображения и текст)
import requests  # Библиотека для отправки HTTP-запросов
import io  # Библиотека для работы с потоками ввода-вывода
from PIL import Image  # Библиотека для обработки изображений
import os  # Библиотека для работы с операционной системой и файлами
from dotenv import load_dotenv  # Библиотека для загрузки переменных окружения из файла .env

# Загрузка переменных окружения из файла .env
# Это позволяет хранить секретные ключи и настройки в отдельном файле, а не в коде
load_dotenv()


def record_or_load_audio(audio_path=None, record_duration=5):
    """
    Функция либо загружает аудиофайл по указанному пути, либо записывает аудио с микрофона

    Параметры:
    - audio_path: путь к аудиофайлу (если None, будет записано новое аудио)
    - record_duration: длительность записи в секундах

    Возвращает:
    - путь к аудиофайлу
    """
    if audio_path and os.path.exists(audio_path):
        print(f"Using existing audio file: {audio_path}")
        return audio_path

    # Если путь не указан или файл не существует, пробуем записать аудио
    print(f"Recording {record_duration} seconds of audio...")
    try:
        import sounddevice as sd  # Библиотека для записи звука
        import soundfile as sf  # Библиотека для сохранения звуковых файлов
        import tempfile  # Библиотека для создания временных файлов

        # Запись аудио
        sample_rate = 16000  # Частота дискретизации (количество образцов в секунду)
        recording = sd.rec(int(record_duration * sample_rate),
                           samplerate=sample_rate, channels=1)  # Запись одноканального аудио
        sd.wait()  # Ожидание завершения записи

        # Сохранение во временный файл
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(temp_file, recording, sample_rate)
        print(f"Audio saved to: {temp_file}")
        return temp_file

    except Exception as e:
        print(f"Error recording audio: {e}")
        print("Please provide an audio file path instead.")
        exit(1)  # Выход из программы с кодом ошибки 1


def transcribe_with_whisper(audio_path):
    """
    Функция транскрибирует (преобразует речь в текст) аудиофайл с помощью модели Whisper

    Параметры:
    - audio_path: путь к аудиофайлу

    Возвращает:
    - текстовая расшифровка аудио
    """
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # Загрузка базовой модели Whisper

    print(f"Transcribing audio: {audio_path}")
    result = model.transcribe(audio_path)  # Преобразование аудио в текст

    transcript = result["text"]  # Извлечение текста из результата
    print(f"Transcript: {transcript}")
    return transcript


def generate_image_with_stable_diffusion(prompt):
    """
    Функция генерирует изображение на основе текстового описания с помощью модели Stable Diffusion

    Параметры:
    - prompt: текстовое описание желаемого изображения

    Возвращает:
    - сгенерированное изображение (объект PIL.Image)
    """
    hf_token = os.getenv("HF_TOKEN")  # Получение токена HuggingFace из переменных окружения
    if not hf_token:
        hf_token = input("Enter your HuggingFace token: ")  # Запрос токена у пользователя, если он не найден

    print(f"Generating image for prompt: '{prompt}'")
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {hf_token}"}  # Заголовок с токеном авторизации
    payload = {"inputs": prompt}  # Данные запроса - текстовое описание

    # Отправка API запроса
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:  # Проверка успешности запроса
        print(f"Error: {response.status_code}, {response.text}")
        return None

    # Преобразование ответа в изображение
    image = Image.open(io.BytesIO(response.content))  # Создание объекта изображения из байтового потока
    image_path = "generated_image.png"  # Путь для сохранения изображения
    image.save(image_path)  # Сохранение изображения на диск
    print(f"Image saved to: {image_path}")
    return image


def analyze_image_with_clip(image):
    """
    Функция анализирует изображение с помощью модели CLIP

    Параметры:
    - image: изображение для анализа (объект PIL.Image)

    Возвращает:
    - top_category: наиболее вероятная категория изображения
    - confidence: уровень уверенности в определении категории (в процентах)
    """
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Выбор устройства для вычислений (GPU или CPU)
    model, preprocess = clip.load("ViT-B/32", device=device)  # Загрузка модели CLIP и функции предобработки

    # Подготовка изображения для CLIP
    image_input = preprocess(image).unsqueeze(0).to(
        device)  # Применение предобработки и перемещение на нужное устройство

    # Определение категорий для проверки
    categories = [
        "a photograph", "a painting", "a digital artwork",  # Типы изображений
        "a landscape", "a portrait", "an abstract image",  # Стили изображений
        "animals", "people", "buildings", "nature",  # Объекты на изображении
        "daytime", "nighttime", "indoor", "outdoor"  # Время суток и местоположение
    ]

    # Кодирование текстовых описаний
    text_tokens = clip.tokenize(categories).to(
        device)  # Преобразование текста в токены и перемещение на нужное устройство

    # Получение предсказаний
    with torch.no_grad():  # Отключение расчета градиентов для экономии памяти
        image_features = model.encode_image(image_input)  # Извлечение признаков из изображения
        text_features = model.encode_text(text_tokens)  # Извлечение признаков из текста

        # Нормализация признаков для корректного сравнения
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Расчет степени сходства между изображением и текстовыми категориями
        similarity = (100.0 * image_features @ text_features.T).softmax(
            dim=-1)  # Матричное умножение для подсчета сходства
        values, indices = similarity[0].topk(5)  # Получение 5 наиболее вероятных категорий

    # Вывод 5 наиболее вероятных совпадений
    print("\nImage Analysis Results:")
    for value, index in zip(values, indices):
        print(f"{categories[index]:>16s}: {100 * value.item():.2f}%")

    # Возврат наиболее вероятной категории
    top_category = categories[indices[0]]
    return top_category, values[0].item()  # Возвращаем лучшую категорию и процент уверенности


def main():
    """
    Основная функция, которая объединяет все этапы обработки
    """
    print("=== Speech to Image with Description Pipeline ===")

    # Шаг 1: Получение или запись аудио
    audio_path = record_or_load_audio(record_duration=5)

    # Шаг 2: Транскрибирование аудио в текст с помощью Whisper
    transcript = transcribe_with_whisper(audio_path)

    # Шаг 3: Генерация изображения из текста с помощью Stable Diffusion
    image = generate_image_with_stable_diffusion(transcript)
    if image is None:
        print("Failed to generate image. Exiting.")
        return

    # Шаг 4: Анализ изображения с помощью CLIP
    top_category, confidence = analyze_image_with_clip(image)

    # Шаг 5: Вывод сводной информации
    print("\n=== Pipeline Summary ===")
    print(f"Speech input transcribed to: '{transcript}'")
    print(f"Generated image saved as: 'generated_image.png'")
    print(f"CLIP analysis: This appears to be {top_category} ({100 * confidence:.2f}% confidence)")

    # Очистка временного аудиофайла, если он был записан
    if audio_path.endswith(".wav") and "tmp" in audio_path:
        try:
            os.remove(audio_path)  # Удаление временного файла
            print(f"Temporary audio file removed: {audio_path}")
        except:
            pass


if __name__ == "__main__":
    # Этот блок выполняется только при непосредственном запуске файла
    # (не при импорте файла как модуля)
    main()
