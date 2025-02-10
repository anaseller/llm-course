import os
from dotenv import load_dotenv
from google import genai
import numpy as np

# Загрузка переменных окружения из файла .env
load_dotenv()

# Получение API-ключа из переменной окружения
api_key = os.getenv("GEMINI_API_KEY")

# Инициализация клиента Gemini для работы с API
client = genai.Client(api_key=api_key)


def get_embedding(text):
    """
    Получает embedding (векторное представление) для заданного текста.

    :param text: Строка текста, для которого требуется получить embedding.
    :return: Список чисел, представляющих векторное embedding.
    """

    # Отправка запроса к API для получения векторного представления текста
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text)

    return np.array(response.embeddings[0].values)     # Возвращаем embedding как numpy array


# Получение embedding для заданных текстовых значений
vector_1 = get_embedding("Я люблю программирование.")
vector_2 = get_embedding("Кодинг – это моё хобби.")

# Вывод полученных векторов с поясняющими сообщениями
print("Я люблю программирование:", vector_1)
print("Кодинг – это моё хобби:", vector_2)
