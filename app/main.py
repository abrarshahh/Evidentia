from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.auth import router as auth_router
from app.api.workspaces import router as workspaces_router

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Evidentia — Enterprise Document Analysis, Claim Extraction, and Context Building Platform API",
    debug=settings.DEBUG,
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth_router)
app.include_router(workspaces_router)


@app.get("/health", tags=["Health"])
async def healthcheck():
    """
    Service healthcheck endpoint.
    """
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "environment": settings.ENV,
    }
