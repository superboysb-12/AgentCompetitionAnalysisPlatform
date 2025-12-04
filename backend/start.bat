@echo off
echo ====================================
echo  Starting Backend Services
echo ====================================
echo.

echo [1/3] Starting FastAPI Server...
start "FastAPI Server" cmd /k "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/3] Starting Celery Worker (Crawl Queue)...
start "Celery Crawl Worker" cmd /k "celery -A celery_app worker --loglevel=info --pool=solo -Q crawl_queue"

echo [3/3] Starting Celery Worker (Extraction Queue)...
start "Celery Extraction Worker" cmd /k "celery -A celery_app worker --loglevel=info --pool=solo -Q extraction_queue"

echo.
echo ====================================
echo  Services Started!
echo ====================================
echo  - FastAPI: http://localhost:8000
echo  - Swagger UI: http://localhost:8000/docs
echo  - 3 windows opened, close each to stop
echo ====================================
echo.

pause
