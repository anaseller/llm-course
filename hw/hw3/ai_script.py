
import os
from dotenv import load_dotenv
from google import genai


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


if not api_key:
    print("Ошибка: API-ключ не найден в переменных окружения. Убедитесь, что файл .env существует и содержит GEMINI_API_KEY.")
else:

    client = genai.Client(api_key=api_key)


    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["What is the capital of France?"]
        )

        print("Ответ от Gemini:")
        print(response.text)

    except Exception as e:
        print(f"Произошла ошибка при отправке запроса: {e}")