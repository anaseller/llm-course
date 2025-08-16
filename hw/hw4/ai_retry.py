import os
import google.api_core.exceptions
import numpy as np
from dotenv import load_dotenv
from google import genai
from numpy.linalg import norm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Ошибка: API-ключ не найден в переменных окружения.")
else:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Ошибка при инициализации клиента: {e}")
        client = None

    if client:
        # Обязательная часть: Защита от Rate Limits и обработка таймаутов
        @retry(
            retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted) |
                  retry_if_exception_type(google.api_core.exceptions.InternalServerError) |
                  retry_if_exception_type(google.api_core.exceptions.DeadlineExceeded),
            wait=wait_random_exponential(multiplier=1, min=4, max=10),
            stop=stop_after_attempt(3),
            after=lambda retry_state: print(f"Попытка {retry_state.attempt_number} не удалась. Повторная попытка...")
        )
        def generate_content_with_retry(prompt):
            print(f"Отправка запроса с промптом: '{prompt}'")
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[prompt],
            )
            return response.text

        # Дополнительная часть: Получение эмбеддингов и их сравнение
        def get_embedding(text):
            try:
                response = client.models.embed_content(
                    model="models/embedding-001",
                    contents=[text],
                )
                # извлекаем вектор из атрибута .values
                return response.embeddings[0].values
            except Exception as e:
                print(f"Ошибка при получении эмбеддинга: {e}")
                return None

        def cosine_similarity(v1, v2):
            v1_np = np.array(v1)
            v2_np = np.array(v2)
            return np.dot(v1_np, v2_np) / (norm(v1_np) * norm(v2_np))

        # Запуск основной логики
        if __name__ == "__main__":
            try:
                my_prompt = "Напиши короткий стих о весне."
                result = generate_content_with_retry(my_prompt)
                print("Ответ от модели:\n", result)
            except Exception as e:
                print(f"Запрос не удалось выполнить после нескольких попыток: {e}")

            print("\n--- Дополнительное задание: Работа с эмбеддингами ---")
            text1 = "Python - это язык программирования."
            text2 = "Программирование на Python очень популярно."
            text3 = "Столица Франции - Париж."

            embedding1 = get_embedding(text1)
            embedding2 = get_embedding(text2)
            embedding3 = get_embedding(text3)

            if all([embedding1, embedding2, embedding3]):
                similarity1_2 = cosine_similarity(embedding1, embedding2)
                similarity1_3 = cosine_similarity(embedding1, embedding3)

                print(f"Косинусное сходство между текстом 1 и текстом 2: {similarity1_2:.4f}")
                print(f"Косинусное сходство между текстом 1 и текстом 3: {similarity1_3:.4f}")

            print("\n--- Простой семантический поиск ---")

            documents = {
                "doc1": "Искусственный интеллект меняет мир.",
                "doc2": "Глубокое обучение является важной частью AI.",
                "doc3": "Париж - столица Франции, и его архитектура великолепна.",
                "doc4": "Компьютерное зрение используется в автономных автомобилях."
            }

            document_embeddings = {
                doc_id: get_embedding(text) for doc_id, text in documents.items()
            }

            query = "Как AI помогает в транспорте?"
            query_embedding = get_embedding(query)

            if query_embedding:
                most_similar_doc = None
                max_similarity = -1

                for doc_id, embedding in document_embeddings.items():
                    if embedding:
                        similarity = cosine_similarity(query_embedding, embedding)
                        print(f"Сходство с документом '{doc_id}': {similarity:.4f}")
                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_doc = doc_id

                print("\n----------------------")
                print(f"Самый похожий документ: {documents[most_similar_doc]} (ID: {most_similar_doc})")
                print(f"Косинусное сходство: {max_similarity:.4f}")