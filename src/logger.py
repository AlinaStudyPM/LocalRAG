# src/logger.py
"""
Модуль логирования. Реализация через метакласс LoggableMeta.
"""
import logging
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, Callable, Optional
from functools import wraps

from src.Config import Config

class JSONFormatter(logging.Formatter):
    """Класс для представления логов в формате JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        return json.dumps(log_data, ensure_ascii=False)

def setup_logging(config: Config) -> logging.Logger:
    """Настраивает логгер на основе конфигурации"""
    Path("logs").mkdir(exist_ok=True)

    logger = logging.getLogger("app")
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    logger.handlers.clear()

    if config.LOG_TO_SESSION_FILE:
        # В файл session.log пишем логи заново
        session_handler = logging.FileHandler(
            config.LOG_SESSION_FILE_PATH, 
            encoding='utf-8',
            mode='w'
        )
        session_handler.setFormatter(JSONFormatter())
        session_handler.setLevel(config.LOG_LEVEL)
        logger.addHandler(session_handler)
    if config.LOG_TO_FILE:
        # В файл app.log добавляем новые логи к старым
        file_handler = logging.FileHandler(
            config.LOG_FILE_PATH,
            encoding='utf-8',
            mode='a'
        )
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(config.LOG_LEVEL)
        logger.addHandler(file_handler)
    if config.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        console_handler.setLevel(config.LOG_LEVEL)
        logger.addHandler(console_handler)

    return logger

class LoggableMeta(type):
    """
    Метакласс для автоматического логгирования всех методов класса.
    """
    _loggers: Dict[str, logging.Logger] = {}
    _depth: Dict[str, int] = {}
    _call_stack: list = []

    def __new__(mcs, name, bases, namespace, _log_enabled=True, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if not _log_enabled:
            return cls
        # Создаём логгер для данного класса
        logger = logging.getLogger(f"app.{name}")
        mcs._loggers[name] = logger
        mcs._depth[name] = 0
        # Декорируем все методы этого класса
        for attr_name, attr_value in list(namespace.items()):
            if callable(attr_value) and not isinstance(attr_value, type):
                mcs._wrap_method(cls, attr_name, attr_value, logger, name)
        return cls

    @classmethod
    def _wrap_method(mcs, cls, method_name, method, logger, class_name):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if getattr(method, '_log_disabled', False):
                return method(self, *args, **kwargs)

            mcs._depth[class_name] = mcs._depth.get(class_name, 0) + 1
            depth = mcs._depth[class_name]

            # ---
            call_info = {
                "class": class_name,
                "method": method_name,
                "depth": depth,
                "args": mcs._safe_repr(args),
                "kwargs": mcs._safe_repr(kwargs),
                "call_id": f"{class_name}.{method_name}#{depth}"
            }

            # VS


            func_path = f"{class_name}.{method_name}"
            args_repr = mcs._format_args(args, kwargs)
            # ---

            logger.debug(
                f"{indent}→ {func_path}({args_repr})",
                extra={'extra_data': {
                    'event': 'call',
                    'function': func_path,
                    'args': mcs._format_args_dict(args, kwargs),
                    'depth': LoggableMeta._depth.get(class_name, 0) - 1
                }}
            )
            
            start_time = time.time()
            try:
                result = method(self, *args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.debug(
                    f"{indent}← {func_path} → {mcs._format_result(result)} ({duration_ms:.1f}ms)",
                    extra={'extra_data': {
                        'event': 'return',
                        'function': func_path,
                        'duration_ms': round(duration_ms, 2),
                        'depth': LoggableMeta._depth.get(class_name, 0) - 1
                    }}
                )
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{indent}✗ {func_path} → {type(e).__name__}: {e} ({duration_ms:.1f}ms)",
                    extra={'extra_data': {
                        'event': 'exception',
                        'function': func_path,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'duration_ms': round(duration_ms, 2),
                        'depth': LoggableMeta._depth.get(class_name, 0) - 1
                    }}
                )
                raise

            finally:
                # Уменьшаем глубину
                mcs._depth[class_name] = mcs._depth.get(class_name, 1) - 1
        
        # Сохраняем оригинальный метод для отладки
        wrapper._original_method = method

        # Привязываем обёртку к классу
        setattr(cls, method_name, wrapper)

    @staticmethod
    def _format_args(args, kwargs) -> str:
        """Строковое представление аргументов"""
        parts = []
        if args:
            parts.append(', '.join(repr(a)[:50] for a in args))
        if kwargs:
            parts.append(', '.join(f"{k}={repr(v)}"[:50] for k, v in kwargs.items()))
        return ', '.join(parts) if parts else ""

    @staticmethod
    def _format_args_dict(args, kwargs) -> dict:
        """Словарь с аргументами для JSON"""
        return {f"arg_{i}": a for i, a in enumerate(args, 1)} | kwargs

    @staticmethod
    def _format_func_result(result) -> str:
        """Строковое представление для отображения результата выполнения функции"""
        if result is None:
            return "None"
        if isinstance(result, (str, int, float, bool)):
            return repr(result)[:50]
        if isinstance(result, (list, dict)):
            return f"<{type(result).__name__} len={len(result)}>"
        return f"<{type(result).__name__}>"

def disable_logging(func: Callable) -> Callable:
    """Декоратор для отключения логгирования конкретного метода."""
    func._log_disabled = True  # type: ignore
    return func

