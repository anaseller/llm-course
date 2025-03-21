import time
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from requests import ReadTimeout

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Инициализируем клиент Gemini с таймаутом (в секундах)
timeout_seconds = 10  # Установите желаемое значение таймаута
client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_seconds * 1000))


# Повторяем попытки подключения к серверу при сбое.
# stop_after_attempt(3): останавливаемся после 3-й попытки.
# wait_exponential(multiplier=1, min=2, max=10):
#   - multiplier=1: начальное время ожидания 2 секунды.
#   - min=2: минимальное время ожидания 2 секунды.
#   - max=10: максимальное время ожидания 10 секунд.
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_gemini_response(prompt):
    """
    Отправляет запрос к модели Gemini и возвращает текст ответа.
    Обрабатывает исключение TimeoutError.

    :param prompt: Текст запроса.
    :return: Текст ответа модели или None в случае таймаута.
    """
    # Небольшая задержка для предотвращения слишком частых запросов (настройте при необходимости)
    time.sleep(0.3)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Используемая модель Gemini
            contents=[prompt]
        )
        return response.text  # Возвращаем текст ответа
    except ReadTimeout:
        # Исправлено: выводим значение таймаута в секундах без преобразования
        return f"Запрос к Gemini превысил таймаут ({timeout_seconds} секунд)."
    except Exception as e:
        # Общая обработка исключений для отлова неожиданных ошибок
        return f"Произошла ошибка: {str(e)}"


if __name__ == "__main__":
    response = get_gemini_response("Whats request timeout?")
    if response:
        print(response)
    else:
        print("Не удалось получить ответ от Gemini.")
