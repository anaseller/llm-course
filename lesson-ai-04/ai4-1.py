import time
from google import genai
from dotenv import load_dotenv
import os

# Загружаем переменные окружения из файла .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Инициализируем клиент Gemini один раз вне функции для улучшения производительности
client = genai.Client(api_key=api_key)


def get_gemini_response(prompt):
    """
    Отправляет запрос к модели Gemini и возвращает текст ответа.

    :param prompt: Текст запроса.
    :return: Текст ответа модели.
    """
    time.sleep(0.3)  # Небольшая задержка перед отправкой запроса (можно убрать или изменить)

    # Используем глобальный объект клиента
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
    )

    return response.text


# Пример использования
if __name__ == "__main__":
    response = get_gemini_response("Whats rate limits?")
    print(response)
