# Импортируем модуль genai из библиотеки google
from google import genai

# Создаём объект клиента для взаимодействия с API Gemini, передавая API-ключ
client = genai.Client(api_key="GEMINI_API_KEY")  # Замените "GEMINI_API_KEY" на свой реальный ключ

# Отправляем запрос к модели Gemini 2.0 Flash
response = client.models.generate_content(
   model="gemini-2.0-flash",  # Выбираем модель (flash - облегчённая версия, быстрая, но менее мощная)
   contents=["How does AI work?"]  # Список с текстом запроса (prompt)
)

# Выводим текст ответа модели
print(response.text)
