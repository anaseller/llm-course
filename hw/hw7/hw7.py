import os
from dotenv import load_dotenv
from PIL import Image
from huggingface_hub import InferenceClient

def setup_env():
    """Настройка окружения и проверка наличия токена Hugging Face"""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("No Hugging Face token found. Please set your HF_TOKEN in a .env file.")
        hf_token = input("Or enter your Hugging Face token now: ")
    return hf_token

def generate_image(prompt, token, negative_prompt="", model_id="stabilityai/stable-diffusion-xl-base-1.0", num_inference_steps=30):
    """
    Генерация изображения с помощью Stable Diffusion через API Hugging Face
    :param prompt: Текстовый запрос
    :param token: Ваш токен Hugging Face
    :param negative_prompt: Что исключить из изображения
    :param model_id: ID модели на Hugging Face (например, "stabilityai/stable-diffusion-xl-base-1.0")
    :param num_inference_steps: Количество шагов генерации
    :return: Объект PIL.Image с сгенерированным изображением
    """
    client = InferenceClient(token=token)

    image = client.text_to_image(
        prompt=prompt,
        model=model_id,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps
    )
    return image

def save_image(image, filename):
    """Сохранение изображения"""
    image.save(filename)
    print(f"Image saved as {filename}")

def main():
    """Основная функция для генерации и сохранения изображений"""
    token = setup_env()

    prompts = [
        'A photo of a cat',
        'A beautiful castle on a hill under a starry night, digital art',
        'A futuristic city in the style of cyberpunk, with flying cars and neon lights, highly detailed, cinematic lighting, ultra high resolution',
    ]

    for i, prompt in enumerate(prompts):
        print(f"Generating image for prompt: '{prompt}'")
        try:
            image = generate_image(
                prompt=prompt,
                token=token,
                negative_prompt="blurry, bad quality, distorted, ugly",
                num_inference_steps=25
            )

            filename = f"generated_image_{i + 1}.png"
            save_image(image, filename)

        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")

if __name__ == "__main__":
    main()