"""HTTP route package."""

from src.routes.health_route import (
    router as health_router,
)
from src.routes.index_route import (
    router as index_router,
)
from src.routes.research_route import (
    create_research_router,
)


__all__ = [
    "create_research_router",
    "health_router",
    "index_router",
]