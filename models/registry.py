import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)


class ModelRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._models: Dict[str, Any] = {}
        self._registry_file = Path("./data/models/registry.json")
        self._registry_file.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        self._load_registry()

    def _load_registry(self):
        if self._registry_file.exists():
            try:
                with open(self._registry_file, 'r', encoding='utf-8') as f:
                    self._models = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model registry: {e}")

    def _save_registry(self):
        try:
            with open(self._registry_file, 'w', encoding='utf-8') as f:
                json.dump(self._models, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save model registry: {e}")

    def register(
        self,
        model_id: str,
        model_type: str,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self._models[model_id] = {
            "model_id": model_id,
            "model_type": model_type,
            "model_path": model_path,
            "config": config or {},
            "metadata": metadata or {},
        }
        self._save_registry()
        logger.info(f"Registered model: {model_id} ({model_type})")

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._models.get(model_id)

    def list_models(self, model_type: Optional[str] = None) -> list:
        if model_type:
            return [
                m for m in self._models.values()
                if m.get("model_type") == model_type
            ]
        return list(self._models.values())

    def unregister(self, model_id: str) -> bool:
        if model_id in self._models:
            del self._models[model_id]
            self._save_registry()
            logger.info(f"Unregistered model: {model_id}")
            return True
        return False

    def get_model_hash(self, model_path: str) -> str:
        import hashlib
        hash_md5 = hashlib.md5()
        hash_md5.update(model_path.encode())
        return hash_md5.hexdigest()


model_registry = ModelRegistry()
