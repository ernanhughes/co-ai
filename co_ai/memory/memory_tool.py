from typing import Optional

import psycopg2

from co_ai.memory import (BaseStore, ContextStore, EmbeddingStore,
                          HypothesesStore)
from co_ai.memory.goal_store import GoalStore
from co_ai.memory.prompt_store import PromptStore
from co_ai.memory.report_logger import ReportLogger
from co_ai.memory.score_store import ScoreStore
from co_ai.memory.pipeline_run_store import PipelineRunStore
from co_ai.memory.reflection_delta_store import ReflectionDeltaStore

from co_ai.memory.lookahead_store import LookaheadStore
from co_ai.memory.mrq_store import MRQStore
from psycopg2.extras import DictCursor  # ← This makes rows dict-like

class MemoryTool:
    def __init__(self, cfg, logger=None):
        self._stores = {}
        db_config = cfg.db  # Load DB config from Hydra
        self.conn = psycopg2.connect(
            dbname=db_config.database,
            user=db_config.user,
            password=db_config.password,
            host=db_config.host,
            port=db_config.port,
            cursor_factory=DictCursor,  # ← Very important!
        )
        self.conn.autocommit = True
        self.logger = logger
        self.cfg = cfg  # Store cfg if needed later
        self.db = self.conn

        self.logger = logger

        self.register_store(GoalStore(self.db, logger))
        self.register_store(EmbeddingStore(self.db, cfg.embeddings, logger))
        self.register_store(HypothesesStore(self.db, self.get("embedding"), logger))
        self.register_store(ContextStore(self.db, logger))
        self.register_store(PromptStore(self.db, logger))
        self.register_store(ReportLogger(self.db, logger))
        self.register_store(MRQStore(self.db, cfg.mrq, self.get("embedding"), logger))
        self.register_store(LookaheadStore(self.db, logger))
        self.register_store(ScoreStore(self.db, logger))
        self.register_store(PipelineRunStore(self.db, logger))
        self.register_store(ReflectionDeltaStore(self.db, logger))

        # Register extra pluggable stores
        if cfg.get("extra_stores"):
            for store in cfg.get("extra_stores"):
                self.register_store(store)

    def register_store(self, store: BaseStore):
        if store.name in self._stores:
            raise ValueError(f"A store with name '{store.name}' is already registered.")
        self._stores[store.name] = store
        print(f"Added {store.name} :=> {store}")
        if self.logger:
            self.logger.log("StoreRegistered", {"store": store.name})

    def get(self, name: str) -> Optional[BaseStore]:
        return self._stores.get(name)

    def __getattr__(self, name):
        if name in self._stores:
            return self._stores[name]
        raise AttributeError(f"'MemoryTool' has no store named '{name}'")