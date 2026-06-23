import json
import logging
import pytest
from config.settings import Settings
import observability.logger as logger_module
from observability.logger import configure_logging, log_event, new_trace_id


@pytest.fixture(autouse=True)
def reset_cogdoc_logging():
    yield
    logger = logging.getLogger("cogdoc")
    for handler in list(logger.handlers):
        if getattr(handler, logger_module._HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()
    logger_module._CONFIGURED_SIGNATURE = None


def test_json_logging_writes_trace_fields(tmp_path):
    log_path = tmp_path / "cogdoc.jsonl"
    settings = Settings(cogdoc_log_file=str(log_path), cogdoc_log_to_console=False)
    configure_logging(settings)

    state = {"request_id": "req-1", "trace_id": "trace-1"}
    log_event("test", "demo_event", state, node_name="router", count=2)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "demo_event"
    assert payload["request_id"] == "req-1"
    assert payload["trace_id"] == "trace-1"
    assert payload["node_name"] == "router"
    assert payload["count"] == 2


def test_trace_id_is_unique_hex():
    first = new_trace_id()
    second = new_trace_id()

    assert first != second
    assert len(first) == 32
    int(first, 16)


def test_configure_logging_is_idempotent_for_same_settings(tmp_path):
    log_path = tmp_path / "cogdoc.jsonl"
    settings = Settings(cogdoc_log_file=str(log_path), cogdoc_log_to_console=False)
    configure_logging(settings)
    logger = logging.getLogger("cogdoc")
    first_handlers = list(logger.handlers)

    configure_logging(settings)

    assert logger.handlers == first_handlers


def test_log_event_prefixes_reserved_extra_keys(tmp_path):
    log_path = tmp_path / "cogdoc.jsonl"
    settings = Settings(cogdoc_log_file=str(log_path), cogdoc_log_to_console=False)
    configure_logging(settings)

    log_event(
        "test",
        "reserved_event",
        {"trace_id": "trace-1"},
        name="business-name",
        module="business-module",
        message="business-message",
    )

    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["field_name"] == "business-name"
    assert payload["field_module"] == "business-module"
    assert payload["field_message"] == "business-message"
