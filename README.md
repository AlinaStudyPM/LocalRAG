# RAG-приложение

## О проекте

Приложение позволяет загружать PDF-документы, извлекать из них текст (включая сканы через OCR), преобразовывать текст в векторные представления и использовать локальную языковую модель (LLM) для ответов на вопросы с контекстом из загруженных документов.

**Основные возможности:**
- Многопользовательская система с авторизацией
- Графический интерфейс на базе Flet
- Разделение документов по коллекциям
- Сохранение истории диалогов
- Автоматическое дополнение запросов контекстом из документов (RAG)


## Запуск на Linux Ubuntu/Debian
#### Установка сторонних компонентов
1. Синхронизировать информацию об обновлении пакетов программ:
```bash
sudo apt-get update
```
2. Установить Python
```bash
sudo apt-get install -y python3 python3-venv
```
3. Установить инструменты для работы с файлами 
```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng poppler-utils curl zstd git
```
4. Установить Ollama для запуска LLM:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
5. Загрузить подходящуюю модель LLM, например:
```bash
ollama pull llama3.2
```
#### Запуск кода
6. Склонировать репозиторий
```bash
git clone https://github.com/AlinaStudyPM/7_semester_coursework.git
cd 7_semester_coursework
```
7. Создать виртуальное окружение для проекта и запустить его:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*Замечание:* для выхода из виртуального окружения используется команда `deactivate`. Для повторного запуска окружения достаточно команды `source .venv/bin/activate`.
8. Установить зависимости
```bash
pip install -e .
```
9. Запустить приложение в нужном режиме:
    - Десктопный: `rag-desktop`
    - Браузерны: `rag-web`


## Запуск на Windows
####  Установка сторонних компонентов  
1. Установить менеджер пакетов Scoop для более простой установки других компонентов:
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
```
2. Установить Poppler и Tesseract для OCR (распознавания текста из документов):
```powershell
scoop install poppler tesseract
```
3. Установить необходимые языки для Tesseract. Для этого необходимо перейти в [репозиторий](https://github.com/tesseract-ocr/tessdata/), и скачать файлы `eng.traineddata` и `rus.traineddata`для английского и русского языков соответсвенно. После поместить их в папку `~\scoop\apps\tesseract\current\tessdata`. После этого команда `tesseract --list-langs` должна показать в терминале `eng` и `rus`.
4. Установить Ollama для запуска LLM.  
```powershell  
irm https://ollama.com/install.ps1 | iex
```  
4. Загрузить подходящуюю модель LLM, например:  
```powershell  
ollama pull llama3.2  
```
#### Запуск кода
5. Склонировать репозиторий
```powershell
git clone https://github.com/AlinaStudyPM/7_semester_coursework.git
cd 7_semester_coursework
```
6. Создать виртуальное окружение для проекта и запустить его:
```powershell
python -m venv .venv
.venv\Scripts\activate
```
*Замечание:* для выхода из виртуального окружения используется команда `deactivate`. Для повторного запуска окружения достаточно команды `.venv\Scripts\activate`.
7. Установить зависимости
```powershell
pip install -e .
```
8. Запустить приложение в нужном режиме:
    - Десктопный: `rag-desktop`
    - Браузерны: `rag-web`


## Конфигурация
1. Настройки по умолчанию указаны в файле `.env.example`. При желании можно создать файл `.env` с собственными настройками.
2. Системный промпт хранится в `config/system_prompt.txt`, его можно отредактировать.
3. Модели LLM автоматически подтягиваются из Ollama. При желании загрузить новые используйте приложение Ollama. Например:
```bash
ollama pull llama3.2
```


## Зависимости
#### Системные пакеты
Эти пакеты необходимо установить вручную согласно инструкции.
| Пакет | Назначение |
|-------|------------|
| ollama | Фреймворк для запуска LLM |
| tesseract-ocr | OCR-движок |
| tesseract-ocr-rus | Русский язык для OCR |
| tesseract-ocr-eng | Английский язык для OCR |
| poppler-utils | Работа с PDF |
#### Python-пакеты
Этих пакеты перечислены в `pyproject.toml` и устанавливаются в виртуальном окружении автоматически при выполнении команды `pip install -e .`.
| Пакет | Назначение |
|-------|------------|
| flet | UI-фреймворк |
| chromadb | Векторная БД |
| ollama | Python-клиент для Ollama |
| langchain-text-splitters | Разделение текста |
| requests | HTTP-запросы |
| pdfplumber | Извлечение из PDF |
| pdf2image | PDF → изображения |
| pytesseract | OCR |
| tiktoken | Токенизация |
| bcrypt | Хеширование паролей |
| python-dotenv | Переменные окружения |
