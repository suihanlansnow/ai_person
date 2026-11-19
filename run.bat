@echo off
setlocal

echo ========================================
echo AI Person - Docker Services Manager
echo ========================================
echo.

if "%1"=="" (
    echo Usage: run.bat [command]
    echo.
    echo Commands:
    echo   start      - Start all services
    echo   stop       - Stop all services
    echo   restart    - Restart all services
    echo   build      - Build all Docker images
    echo   logs       - View logs
    echo   status     - Show service status
    echo   clean      - Clean up containers and volumes
    echo.
    echo ========================================
    echo GPU Services (require NVIDIA GPU):
    echo   start-full    - Start with GPU support
    echo ========================================
    exit /b 1
)

set COMMAND=%1

if "%COMMAND%"=="start" (
    echo Starting AI Person services...
    docker-compose up -d
    echo.
    echo Services started. Access the application at: http://localhost:8000
    echo Grafana dashboard at: http://localhost:3000
    echo Prometheus at: http://localhost:9090
) else if "%COMMAND%"=="start-full" (
    echo Starting AI Person services with GPU support...
    docker-compose up -d
    echo.
    echo Services started with GPU support.
) else if "%COMMAND%"=="stop" (
    echo Stopping AI Person services...
    docker-compose stop
) else if "%COMMAND%"=="restart" (
    echo Restarting AI Person services...
    docker-compose restart
) else if "%COMMAND%"=="build" (
    echo Building Docker images...
    docker-compose build --no-cache
) else if "%COMMAND%"=="logs" (
    docker-compose logs -f
) else if "%COMMAND%"=="status" (
    docker-compose ps
) else if "%COMMAND%"=="clean" (
    echo Cleaning up containers and volumes...
    docker-compose down -v
    echo Cleanup complete.
) else (
    echo Unknown command: %COMMAND%
    exit /b 1
)

echo.
echo Done.