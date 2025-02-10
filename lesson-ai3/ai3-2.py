from google import genai
from dotenv import load_dotenv
import os

# Загружаем переменные окружения из файла .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Создаём объект клиента для взаимодействия с API Gemini, передавая API-ключ
client = genai.Client(api_key=api_key)

# Отправляем запрос к модели Gemini 2.0 Flash
response = client.models.generate_content(
    model="gemini-2.0-flash",  # Выбираем модель (flash - облегчённая версия, быстрая, но менее мощная)
    contents=["How to keep API key secret?"]  # Список с текстом запроса (prompt)
)

# Выводим текст ответа модели
print(response.text)
