import os
import faiss
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
# print("Я люблю программирование:", vector_1)
# print("Кодинг – это моё хобби:", vector_2)


# ---  Секция семантического поиска  ---

# 1. Создание набора текстов для поиска
texts_to_index = [
    "Кошка сидит на окне",
    "Собака играет в парке",
    "Ананас растет в тропиках",
    "Кот спит на диване",
    "Пес лает на почтальона",
    "Фрукт ананас очень вкусный",
    "Домашняя кошка любит ласку",
    "Верный пес охраняет дом",
    "Спелый ананас полон витаминов",
    "Кошки любят рыбу и мясо",
    "Основной рацион кошек - это белок",
    "Чем кормить котенка?",
    "Лучший корм для кошек - сбалансированный",
    "Коты едят сухой и влажный корм",
    "Нельзя кормить кошку шоколадом",
    "Молоко не всегда полезно для кошек",
    "Я обожаю программировать.",
    "Программирование – это то, что мне очень нравится.",
    "Меня увлекает разработка программного обеспечения.",
    "Я испытываю страсть к написанию кода.",
    "Программирование приносит мне огромное удовольствие.",
    "Мне интересно заниматься программированием.",
    "Моё хобби – это кодинг.",
    "В свободное время я занимаюсь программированием.",
    "Кодинг – это моё любимое увлечение.",
    "Я увлекаюсь кодингом на досуге.",
    "Программирование – это моё хобби и страсть.",
    "Когда есть свободное время, я кодирую.",
    "Кодинг - это моё хобби, которым я наслаждаюсь."
]

# 2. Получение embedding для каждого текста и сохранение их в списке
embeddings_list = [get_embedding(text) for text in texts_to_index]
embeddings_array = np.array(embeddings_list)    # Преобразуем в numpy array для FAISS

# 3. Создание FAISS индекса
dimension = embeddings_array.shape[1]   # Размерность embedding
index = faiss.IndexFlatL2(dimension)    # Используем IndexFlatL2 для L2 дистанции (евклидова)
index.add(embeddings_array)             # Добавляем embeddings в индекс


# 4. Функция для выполнения семантического поиска
def semantic_search(query, index, texts, k=2):
    """
    Выполняет семантический поиск по индексу FAISS.

    :param query: Поисковый запрос (строка).
    :param index: FAISS индекс.
    :param texts: Список текстов, которые были проиндексированы.
    :param k: Количество ближайших соседей для поиска (по умолчанию 2).
    :return: Список из k наиболее релевантных текстов.
    """
    query_embedding = get_embedding(query).reshape(1, -1)   # Получаем embedding для запроса и меняем размерность
    D, I = index.search(query_embedding, k)                 # Ищем k ближайших соседа
    results = [texts[i] for i in I[0]]                      # Получаем тексты по индексам
    return results


# 5. Пример использования семантического поиска
search_query = "Питание кошек"
search_results = semantic_search(search_query, index, texts_to_index, k=4)

print("\n--- Результаты семантического поиска ---")
print(f"Запрос: '{search_query}'")
print("Найденные соответствия:")
for result in search_results:
    print(f"- {result}")
