# Импортируем необходимые классы и функции из сторонних библиотек:
from langchain_google_genai import ChatGoogleGenerativeAI  # Класс для работы с моделью генеративного ИИ от Google (в данном случае Gemini).
from langchain_community.tools.tavily_search import TavilySearchResults  # Инструмент для выполнения поисковых запросов через Tavily.
from langchain_core.messages import HumanMessage  # Класс для создания сообщений, имитирующих ввод от человека.
from langgraph.checkpoint.memory import MemorySaver  # Модуль для сохранения состояния (памяти) диалога.
from langgraph.prebuilt import create_react_agent  # Функция для создания агента, использующего паттерн "ReAct" (Reasoning + Acting).
from dotenv import load_dotenv  # Функция для загрузки переменных окружения из файла .env.
import os  # Модуль для работы с операционной системой, в частности, для работы с переменными окружения.

# Загружаем переменные окружения из файла .env
load_dotenv()

# Получаем API-ключ для модели Gemini и Tavily из переменных окружения
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Создаём экземпляр генеративной модели ИИ от Google, используя модель "gemini-2.0-flash"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Инициализируем объект для сохранения состояния диалога (память агента)
memory = MemorySaver()

# Создаём объект инструмента для поиска, ограничивая число результатов до 2
search = TavilySearchResults(max_results=2)

# Собираем все инструменты в список. Здесь используется только инструмент поиска.
tools = [search]

# Создаём агента, который будет обрабатывать запросы.
# Для этого передаём ему генеративную модель, список инструментов и объект для сохранения памяти.
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Создаём конфигурационный словарь с настройками, например, идентификатором потока.
config = {"configurable": {"thread_id": "abc123"}}

# Первый цикл: отправляем сообщение "hi im bob! and i live in berlin" агенту и выводим промежуточные шаги выполнения.
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in berlin")]},
    config,
    stream_mode="values",
):
    # Выводим последнее сообщение из текущего шага в красивом формате.
    step["messages"][-1].pretty_print()

# Второй цикл: отправляем сообщение "whats the weather where I live?" агенту.
# Агент использует накопленную память и, возможно, инструмент поиска, чтобы сформировать ответ.
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]},
    config,
    stream_mode="values",
):
    # Аналогично, выводим последнее сообщение из каждого шага.
    step["messages"][-1].pretty_print()
