# Импорт необходимых библиотек
import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    # prepare_model_for_kbit_training, # Эту строку можно удалить
    TaskType
)
from tqdm import tqdm
import warnings
from dotenv import load_dotenv

# ===================== ЧАСТЬ 1: ПОДГОТОВКА ДАННЫХ =====================
# Создаем небольшой датасет о вымышленной компании
company_info = {
    "name": "Innovate AI",
    "location": "San Francisco, CA",
    "ceo": "Alex Chen",
    "products": "AI-powered data analysis tools, custom machine learning models, and smart automation software.",
    "mission": "To empower businesses with cutting-edge AI solutions."
}

# Шаблон, который используется для создания обучающих примеров.
# Это формат, который модель будет ожидать во время обучения.
instruction_template = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)

# Создаем обучающие примеры, используя данные о компании
training_examples = [
    {"instruction": f"What is {company_info['name']}?",
     "response": f"{company_info['name']} is a company located in {company_info['location']}."},
    {"instruction": f"Who is the CEO of {company_info['name']}?",
     "response": f"The CEO of {company_info['name']} is {company_info['ceo']}."},
    {"instruction": f"What products does {company_info['name']} offer?",
     "response": f"{company_info['name']} offers {company_info['products']}."},
    {"instruction": f"What is the mission of {company_info['name']}?",
     "response": f"The mission of {company_info['name']} is {company_info['mission']}."},
]

# Преобразуем данные в датафрейм, а затем в Hugging Face Dataset
df = pd.DataFrame(training_examples)
dataset = Dataset.from_pandas(df)

# Функция для токенизации и форматирования данных
def format_examples(examples):
    formatted_texts = []
    for instruction, response in zip(examples['instruction'], examples['response']):
        text = instruction_template.format(instruction=instruction, response=response)
        formatted_texts.append(text)
    return {'text': formatted_texts}

dataset = dataset.map(format_examples, batched=True, remove_columns=['instruction', 'response'])

# ===================== ЧАСТЬ 2: АВТОРИЗАЦИЯ И НАСТРОЙКА МОДЕЛИ =====================
# Загрузка токена из .env файла
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Проверяем, что токен был найден
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Please set HF_TOKEN in your .env file.")

# Игнорируем предупреждения от Transformers
warnings.filterwarnings("ignore", category=UserWarning)

# Выбираем модель, которую будем дообучать
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Определяем устройство для обучения
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загружаем предобученную модель и токенизатор
# Загрузка в 8-битном режиме убрана
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Настраиваем токенизатор
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Устанавливаем padding_side

# ===================== ЧАСТЬ 3: FINE-TUNING С PEFT/LORA =====================
# Строка prepare_model_for_kbit_training(model) удалена
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Выводим количество обучаемых параметров

# Аргументы для обучения
training_args = TrainingArguments(
    output_dir="./lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    fp16=False,
    optim="paged_adamw_8bit",
)

# Создаем Trainer для обучения модели
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# Запускаем обучение
print("\nStarting fine-tuning...")
trainer.train()
print("Fine-tuning finished.")

# ===================== ЧАСТЬ 4: ОЦЕНКА МОДЕЛИ =====================
# Функция для генерации ответа от дообученной модели
def generate_response(question, model, tokenizer, device):
    inputs = tokenizer(instruction_template.format(instruction=question, response=""),
                       return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    # Декодируем сгенерированные токены в читаемый текст
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Извлекаем только ту часть, которая идет после метки "### Response:"
    response = response.split("### Response:")[1].strip()
    return response

# Тестовые вопросы для проверки работы дообученной модели
test_questions = [
    f"What is {company_info['name']}?",
    f"Who is the CEO of {company_info['name']}?",
    f"What products does {company_info['name']} offer?",
    "What was the company's revenue in 2020?"  # Этот вопрос демонстрирует обработку запроса, если информации нет
]

print("\nTesting the fine-tuned model:")
for question in test_questions:
    response = generate_response(question, model, tokenizer, device)
    print(f"\nQ: {question}")
    print(f"A: {response}")
    print("-" * 50)

# ===================== ЧАСТЬ 6: ПРОСТОЙ ИНТЕРФЕЙС =====================
# Функция для интерактивного режима, где пользователь может задавать свои вопросы
def interactive_qa():
    print("\n" + "=" * 50)
    print(f"Ask questions about {company_info['name']}")
    print("Type 'exit' to quit")
    print("=" * 50)

    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() in ['exit', 'quit', 'q']:
            break

        response = generate_response(user_question, model, tokenizer, device)
        print(f"\nAI Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    interactive_qa()
