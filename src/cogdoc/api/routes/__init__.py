from cogdoc.api.routes.chat import router as chat_router
from cogdoc.api.routes.documents import router as documents_router
from cogdoc.api.routes.feedback import router as feedback_router
from cogdoc.api.routes.health import router as health_router
from cogdoc.api.routes.traces import router as traces_router

__all__ = [
    "chat_router",
    "documents_router",
    "feedback_router",
    "health_router",
    "traces_router",
]
