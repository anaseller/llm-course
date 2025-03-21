# Импортируем стандартный модуль для работы с датой и временем.
from datetime import datetime
# Импортируем класс Tool для создания инструментов.
from langchain_core.tools import Tool
# Импортируем класс для работы с генеративной моделью ИИ от Google (в данном случае используется модель Gemini).
from langchain_google_genai import ChatGoogleGenerativeAI
# Импортируем инструмент для выполнения поисковых запросов через сервис Tavily.
from langchain_community.tools.tavily_search import TavilySearchResults
# Импортируем класс для создания сообщений от пользователя.
from langchain_core.messages import HumanMessage
# Импортируем модуль для сохранения состояния диалога (память агента).
from langgraph.checkpoint.memory import MemorySaver
# Импортируем функцию для создания агента по принципу ReAct (сочетание рассуждения и действий).
from langgraph.prebuilt import create_react_agent
# Импортируем функцию для загрузки переменных окружения из файла .env.
from dotenv import load_dotenv
# Импортируем модуль для работы с операционной системой, в частности для управления переменными окружения.
import os

# Загружаем переменные окружения из файла .env.
load_dotenv()

# Получаем API-ключ для модели Gemini и устанавливаем его в переменную окружения, которую использует библиотека.
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
# Получаем API-ключ для сервиса Tavily и устанавливаем его в переменную окружения.
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Создаём экземпляр генеративной модели ИИ от Google с использованием модели "gemini-2.0-flash".
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Инициализируем объект для сохранения состояния диалога (память агента), что позволяет агенту запоминать предыдущие сообщения.
memory = MemorySaver()

# Создаём инструмент для поиска с ограничением числа результатов до 2.
search = TavilySearchResults(max_results=2)


# Функция для получения текущей даты и времени в формате ISO.
def get_current_date(*args, **kwargs):
    return datetime.now().isoformat()


# Создаём инструмент, оборачивая функцию get_current_date.
# Он будет доступен агенту для вызова при необходимости узнать текущее время.
date_tool = Tool(
    name="Datetime",
    func=get_current_date,
    description="Returns current datetime in ISO format."
)

# Собираем все инструменты в список. Здесь используются как инструмент поиска, так и инструмент получения даты.
tools = [search, date_tool]

# Создаём агента, который будет обрабатывать входящие сообщения.
# Передаём ему генеративную модель, список инструментов и объект для сохранения памяти.
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Создаём конфигурационный словарь с настройками, например, с уникальным идентификатором потока (thread_id).
config = {"configurable": {"thread_id": "abc123"}}

# Первый цикл: отправляем агенту сообщение, представляющее пользователя.
# Здесь сообщение содержит приветствие и информацию о пользователе.
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in berlin")]},
    config,
    stream_mode="values",
):
    # Выводим последнее сообщение из текущего шага в формате, удобном для чтения.
    step["messages"][-1].pretty_print()

# Второй цикл: отправляем агенту запрос для получения сегодняшней даты.
# В данном случае запрос: "whats the date for today?"
# Агент может использовать инструмент получения даты.
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the date for today?")]},
    config,
    stream_mode="values",
):
    # Аналогично выводим последнее сообщение из каждого шага выполнения.
    step["messages"][-1].pretty_print()
