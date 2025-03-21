import os
import torch  # Библиотека для работы с нейронными сетями
import whisper  # Библиотека от OpenAI для распознавания речи
import tempfile  # Для создания временных файлов
from datetime import datetime  # Для работы с датой и временем
from pathlib import Path  # Для работы с путями файлов
import argparse  # Для разбора аргументов командной строки
import sys  # Для работы с системными функциями


def setup_whisper():
    """
    Устанавливает и настраивает Whisper, если это необходимо.
    Возвращает доступное устройство (CUDA или CPU).

    CUDA - технология для использования видеокарты (GPU) для вычислений
    CPU - центральный процессор компьютера
    """
    # Проверяем, установлен ли whisper, если нет - устанавливаем его
    try:
        import whisper  # Пытаемся импортировать библиотеку
    except ImportError:  # Если библиотека не установлена, возникнет ошибка
        print("Installing OpenAI Whisper...")  # Сообщаем о начале установки
        os.system("pip install -U openai-whisper")  # Запускаем команду установки через pip
        try:
            import whisper  # Пробуем снова импортировать после установки
        except ImportError:  # Если установка не удалась
            print("Failed to install whisper. Please install manually:")  # Выводим сообщение о неудаче
            print("pip install -U openai-whisper")  # Предлагаем установить вручную
            sys.exit(1)  # Завершаем программу с кодом ошибки 1

    # Определяем устройство для вычислений
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Если доступен GPU, используем его, иначе CPU
    print(f"Using device: {device}")  # Выводим информацию об используемом устройстве

    return device  # Возвращаем определенное устройство


def load_model(model_size="tiny", device=None):
    """
    Загружает модель Whisper.

    Аргументы:
        model_size (str): Размер модели. Варианты: "tiny", "base", "small", "medium", "large"
        device (str): Устройство для использования (cuda или cpu)

    Возвращает:
        Загруженную модель

    Примечание: Чем больше модель, тем точнее распознавание, но требуется больше памяти и времени
    """
    print(f"Loading Whisper {model_size} model...")  # Сообщаем о начале загрузки модели
    model = whisper.load_model(model_size, device=device)  # Загружаем модель указанного размера
    print(f"Model loaded successfully.")  # Сообщаем об успешной загрузке
    return model  # Возвращаем загруженную модель


def transcribe_audio_file(model, audio_path, language=None):
    """
    Транскрибирует (преобразует речь в текст) аудиофайл с помощью Whisper.

    Аргументы:
        model: Модель Whisper
        audio_path (str): Путь к аудиофайлу
        language (str, опционально): Код языка (например, "en", "fr", "ja", "ru" и т.д.)

    Возвращает:
        dict: Результат транскрипции (словарь с текстом и доп. информацией)
    """
    print(f"Transcribing: {audio_path}")  # Сообщаем о начале транскрипции

    # Устанавливаем параметры транскрипции
    options = {}  # Создаем пустой словарь для параметров
    if language:  # Если указан язык
        options["language"] = language  # Добавляем его в параметры

    # Транскрибируем аудиофайл
    result = model.transcribe(audio_path, **options)  # **options передает все параметры из словаря
    return result  # Возвращаем результат транскрипции


def save_transcription(result, output_file=None):
    """
    Сохраняет результат транскрипции в файл.

    Аргументы:
        result (dict): Результат транскрипции от Whisper
        output_file (str, опционально): Путь для сохранения результата

    Возвращает:
        str: Путь к сохраненному файлу
    """
    # Генерируем имя выходного файла, если оно не указано
    if not output_file:  # Если имя файла не указано
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S")  # Создаем метку времени в формате ГодМесяцДень_ЧасМинутаСекунда
        output_file = f"transcription_{timestamp}.txt"  # Формируем имя файла с временной меткой

    # Сохраняем транскрипцию
    with open(output_file, "w", encoding="utf-8") as f:  # Открываем файл для записи с кодировкой UTF-8
        f.write(result["text"])  # Записываем текст из результата в файл

    print(f"Transcription saved to: {output_file}")  # Сообщаем о сохранении транскрипции
    return output_file  # Возвращаем путь к сохраненному файлу


def record_audio(duration=5, sample_rate=16000, channels=1):
    """
    Записывает аудио с микрофона.

    Аргументы:
        duration (int): Длительность записи в секундах
        sample_rate (int): Частота дискретизации (качество звука)
        channels (int): Количество каналов (1=моно, 2=стерео)

    Возвращает:
        str: Путь к записанному аудиофайлу
    """
    try:
        # Пробуем импортировать библиотеки для записи звука
        import sounddevice as sd  # Для записи аудио
        import soundfile as sf  # Для сохранения аудио в файл
    except ImportError:  # Если библиотеки не установлены
        print("Installing required packages for audio recording...")  # Сообщаем о начале установки
        os.system("pip install sounddevice soundfile")  # Устанавливаем нужные пакеты
        import sounddevice as sd  # Импортируем после установки
        import soundfile as sf  # Импортируем после установки

    print(f"Recording audio for {duration} seconds...")  # Сообщаем о начале записи
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)  # Начинаем запись
    sd.wait()  # Ждем окончания записи

    # Сохраняем запись во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:  # Создаем временный файл с расширением .wav
        temp_file = f.name  # Запоминаем имя временного файла

    sf.write(temp_file, recording, sample_rate)  # Записываем аудио во временный файл
    print(f"Audio recorded and saved to temporary file: {temp_file}")  # Сообщаем о сохранении аудио

    return temp_file  # Возвращаем путь к временному файлу


def main():
    """
    Основная функция программы.
    Обрабатывает аргументы командной строки и управляет процессом транскрипции.
    """
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Whisper Speech Recognition Tool")
    # Добавляем аргументы командной строки:
    parser.add_argument("--file", type=str, help="Path to audio file for transcription")  # Путь к аудиофайлу
    parser.add_argument("--record", type=int, default=0,
                        help="Record audio for specified seconds")  # Запись аудио на указанное количество секунд
    parser.add_argument("--model", type=str, default="base",  # Размер модели (по умолчанию "base")
                        choices=["tiny", "base", "small", "medium", "large"],  # Доступные размеры моделей
                        help="Whisper model size")
    parser.add_argument("--language", type=str, help="Language code (e.g., 'en', 'fr')")  # Код языка
    parser.add_argument("--output", type=str, help="Output file path")  # Путь к выходному файлу

    args = parser.parse_args()  # Разбираем аргументы командной строки

    # Настройка
    device = setup_whisper()  # Устанавливаем и настраиваем Whisper

    # Загружаем модель
    model = load_model(args.model, device)  # Загружаем модель указанного размера

    # Получаем путь к аудиофайлу
    audio_path = None  # Изначально путь к аудио не задан
    if args.file:  # Если указан аудиофайл
        audio_path = args.file  # Используем указанный файл
    elif args.record > 0:  # Если указано время записи
        audio_path = record_audio(duration=args.record)  # Записываем аудио с микрофона
    else:  # Если не указан ни файл, ни запись
        print("No audio input specified. Use --file or --record")  # Сообщаем, что не указан источник аудио
        print("Example: python whisper_demo.py --file audio.mp3")  # Пример использования с файлом
        print("Example: python whisper_demo.py --record 10")  # Пример использования с записью
        return  # Завершаем функцию

    # Транскрибируем аудио
    result = transcribe_audio_file(model, audio_path, args.language)  # Распознаем речь из аудио

    # Выводим результат транскрипции
    print("\nTranscription:")  # Заголовок для транскрипции
    print(result["text"])  # Выводим распознанный текст

    # Сохраняем транскрипцию, если указан выходной файл
    if args.output:  # Если указан путь для сохранения
        save_transcription(result, args.output)  # Сохраняем транскрипцию в указанный файл

    # Удаляем временный файл, если было выполнено аудио
    if args.record > 0 and audio_path:  # Если была запись
        os.unlink(audio_path)  # Удаляем временный файл
        print(f"Temporary audio file deleted: {audio_path}")  # Сообщаем об удалении


# Точка входа в программу
if __name__ == "__main__":  # Если этот файл запущен напрямую (не импортирован)
    main()  # Вызываем основную функцию
