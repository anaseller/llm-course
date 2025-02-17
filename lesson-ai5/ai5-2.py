import os  # Модуль для работы с операционной системой (например, для работы с переменными окружения).
from dotenv import load_dotenv  # Функция для загрузки переменных окружения из файла .env.
from langchain_community.document_loaders import PyPDFLoader  # Импортируем загрузчик для PDF-файлов.
import asyncio  # Модуль для работы с асинхронным программированием.
from langchain_core.vectorstores import InMemoryVectorStore  # Импортируем класс для создания векторного хранилища в памяти.
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Импортируем класс для генерации эмбеддингов с использованием Google Generative AI.


# Определяем асинхронную функцию для чтения PDF-файла
async def read_pdf():
    # Создаем объект загрузчика для указанного PDF-файла
    loader = PyPDFLoader('../files/The-Old-Man-and-The-Sea-by-Ernest-Hemingway.pdf')    # Нужно вставить свой путь к файлу
    pages = []  # Инициализируем пустой список для хранения страниц из PDF

    # Асинхронно перебираем страницы PDF с помощью метода alazy_load()
    async for page in loader.alazy_load():
        pages.append(page)  # Добавляем каждую страницу в список

    return pages  # Возвращаем список страниц


# Загружаем переменные окружения из файла .env
load_dotenv()

# Получаем API-ключ из переменной окружения "GEMINI_API_KEY"
api_key = os.getenv("GEMINI_API_KEY")

# Если переменная "GOOGLE_API_KEY" не установлена, присваиваем ей значение из GEMINI_API_KEY
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = api_key

# Выводим сообщение о начале чтения PDF-файла
print('Start reading..')
# Запускаем асинхронную функцию read_pdf и сохраняем результат (список страниц)
result = asyncio.run(read_pdf())

# Выводим сообщение о начале формирования эмбеддингов
print('Start embeding..')
# Создаем векторное хранилище, преобразуя документы (страницы PDF) в эмбеддинги.
# Для генерации эмбеддингов используется модель GoogleGenerativeAIEmbeddings с указанной моделью "models/embedding-001".
vector_store = InMemoryVectorStore.from_documents(result, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Выполняем поиск документов по смысловому запросу.
# Функция similarity_search ищет наиболее похожие страницы по заданному запросу.
docs = vector_store.similarity_search("Moment when sharks for the first time attack the fish", k=2)

# Перебираем найденные документы и выводим номер страницы и содержание.
for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content}\n')
