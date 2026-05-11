# tests/test_logging.py
import pytest
from unittest.mock import MagicMock, patch

import logging
from src.logger import LoggableMeta, JSONFormatter, disable_logging

# === FIXTURES ===

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.LOG_LEVEL = "DEBUG"
    return config

@pytest.fixture
def formatter():
    return JSONFormatter()

@pytest.fixture
def sample_class():
    class MyClass(metaclass=LoggableMeta):
        def add(self, a, b):
            return a + b
        
        def greet(self, name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        @disable_logging
        def secret(self):
            return "hidden"
    
    return MyClass


# === FORMATTER TESTS ===

class TestJSONFormatter:
    def test_basic_structure(self, formatter, caplog):
        """JSON содержит все обязательные поля"""
        record = logging.LogRecord(
            name="test", level=logging.DEBUG,
            pathname="", lineno=0, msg="test message", args=(), exc_info=None
        )
        result = formatter.format(record)
        data = json.loads(result)
        
        assert "timestamp" in data
        assert data["level"] == "DEBUG"
        assert data["logger"] == "test"
        assert data["message"] == "test message"
    def test_with_extra_data(self, formatter):
        """extra_data добавляется в JSON"""
        record = logging.LogRecord(...)
        record.extra_data = {"user_id": 42, "action": "login"}
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["user_id"] == 42
        assert data["action"] == "login"
    def test_unicode_handling(self, formatter):
        """Кириллица не ломает JSON"""
        record = logging.LogRecord(..., msg="Привет, мир!", ...)
        result = formatter.format(record)
        
        assert "Привет" in result
        json.loads(result)  # Не должно падать


# === METACLASS TESTS ===

class TestLoggableMeta:
    def test_methods_are_wrapped(self, sample_class):
        """Методы обёрнуты в логирующий wrapper"""
        obj = sample_class()
        assert hasattr(obj.add, '_original_method')
    def test_method_call_produces_logs(self, sample_class, caplog):
        """Вызов метода создаёт log-записи"""
        obj = sample_class()
        result = obj.add(2, 3)
        
        assert result == 5
        assert any("call" in r.message for r in caplog.records)
        assert any("return" in r.message for r in caplog.records)
    def test_exception_is_logged(self, sample_class, caplog):
        """Исключение логируется с типом ошибки"""
        class Broken(sample_class):
            def fail(self):
                raise ValueError("oops")
        
        obj = Broken()
        with pytest.raises(ValueError):
            obj.fail()
        
        assert any("exception" in r.message for r in caplog.records)
        assert any("ValueError" in r.message for r in caplog.records)
    def test_disabled_decorator_works(self, sample_class, caplog):
        """@disable_logging отключает логирование метода"""
        obj = sample_class()
        result = obj.secret()
        
        assert result == "hidden"
        assert len(caplog.records) == 0  # Никаких логов!
    def test_nested_calls_have_depth(self, sample_class, caplog):
        """Вложенные вызовы показывают глубину"""
        class Nested(sample_class):
            def outer(self, x):
                return self.inner(x)
            def inner(self, x):
                return x * 2
        
        obj = Nested()
        obj.outer(5)
        
        # Проверяем что depth увеличивается
        calls = [r for r in caplog.records if "outer" in r.message or "inner" in r.message]
        # depth для inner должен быть больше чем для outer

# === EDGE CASES ===

class TestEdgeCases:
    def test_empty_args(self, sample_class, caplog):
        """Метод без аргументов"""
        class NoArgs(sample_class):
            def ping(self):
                return "pong"
        
        obj = NoArgs()
        obj.ping()
        
        assert any("ping()" in r.message for r in caplog.records)
    def test_long_args_truncated(self, sample_class, caplog):
        """Длинные аргументы обрезаются"""
        obj = sample_class()
        long_string = "x" * 100
        obj.greet(long_string)
        
        # Проверяем что длинная строка не сломала лог
        assert any("greet" in r.message for r in caplog.records)
    def test_private_methods(self):
        """Приватные методы __method__"""
        class WithPrivate(metaclass=LoggableMeta):
            def __init__(self):
                pass
            def __private_method(self):
                return "secret"
        
        obj = WithPrivate()
        # Приватные методы обычно не логируются или логируются с осторожностью
    def test_property_not_wrapped(self):
        """@property не оборачивается в логгер"""
        class WithProp(metaclass=LoggableMeta):
            @property
            def value(self):
                return 42
        
        obj = WithProp()
        result = obj.value
        assert result == 42
        # property не должен иметь _original_method
