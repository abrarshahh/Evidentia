import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import setup_logging, set_log_context, clear_log_context
from app.auth.security import decode_token
from app.api.auth import router as auth_router
from app.api.workspaces import router as workspaces_router
from app.api.documents import router as documents_router
from app.api.claims import router as claims_router
from app.api.context import router as context_router
from app.api.analyses import router as analyses_router

# Initialize Centralized Logging System
setup_logging(level=logging.INFO if not settings.DEBUG else logging.DEBUG)
logger = logging.getLogger(__name__)

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


@app.middleware("http")
async def log_context_middleware(request: Request, call_next):
    """
    Middleware to automatically extract Bearer token user & analysis context for log enrichment.
    """
    clear_log_context()

    # Extract user identity from Bearer Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        try:
            token_data = decode_token(token)
            set_log_context(user_id=token_data.email or token_data.user_id)
        except Exception:
            pass

    # Extract analysis ID from query parameters or custom header
    analysis_id = request.query_params.get("analysis_id") or request.headers.get("X-Analysis-ID")
    if analysis_id:
        set_log_context(analysis_id=analysis_id)

    response = await call_next(request)
    return response


# Include API Routers
app.include_router(auth_router)
app.include_router(workspaces_router)
app.include_router(documents_router)
app.include_router(claims_router)
app.include_router(context_router)
app.include_router(analyses_router)


@app.get("/health", tags=["Health"])
async def healthcheck():
    """
    Service healthcheck endpoint.
    """
    logger.info("Healthcheck endpoint pinged.")
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "environment": settings.ENV,
    }

