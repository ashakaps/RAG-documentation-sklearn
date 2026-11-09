"""Общие настройки тестов.

`app.config.Settings` требует `LLM_API_KEY` при импорте. Тестам реальный ключ
не нужен (LLM замокана), поэтому подставляем заглушку до импорта `app`:
так pytest работает и на чистой машине, и в CI без секретов.
"""

import os

os.environ.setdefault("LLM_API_KEY", "test-key")
