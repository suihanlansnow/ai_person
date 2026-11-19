#!/bin/bash

echo "========================================"
echo "AI Person - Docker Services Manager"
echo "========================================"
echo ""

show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start      - Start all services"
    echo "  stop       - Stop all services"
    echo "  restart    - Restart all services"
    echo "  build      - Build all Docker images"
    echo "  rebuild    - Rebuild without cache"
    echo "  logs       - View logs (all services)"
    echo "  logs [svc] - View logs for specific service"
    echo "  status     - Show service status"
    echo "  ps         - Show running containers"
    echo "  clean      - Clean up containers and volumes"
    echo "  health     - Check health of all services"
    echo ""
    echo "========================================"
    echo "GPU Services (require NVIDIA GPU + nvidia-docker):"
    echo "  start-gpu  - Start with GPU support"
    echo "========================================"
}

COMMAND=${1:-""}

case "$COMMAND" in
    start)
        echo "Starting AI Person services..."
        docker-compose up -d
        echo ""
        echo "Services started successfully!"
        echo "  API Gateway: http://localhost:8000"
        echo "  Grafana:    http://localhost:3000"
        echo "  Prometheus: http://localhost:9090"
        ;;
    start-gpu)
        echo "Starting AI Person services with GPU support..."
        docker-compose up -d
        echo ""
        echo "Services started with GPU support!"
        ;;
    stop)
        echo "Stopping AI Person services..."
        docker-compose stop
        echo "Services stopped."
        ;;
    restart)
        echo "Restarting AI Person services..."
        docker-compose restart
        echo "Services restarted."
        ;;
    build)
        echo "Building Docker images..."
        docker-compose build
        echo "Build complete."
        ;;
    rebuild)
        echo "Rebuilding Docker images without cache..."
        docker-compose build --no-cache
        echo "Rebuild complete."
        ;;
    logs)
        SERVICE=${2:-""}
        if [ -z "$SERVICE" ]; then
            docker-compose logs -f
        else
            docker-compose logs -f "$SERVICE"
        fi
        ;;
    status|ps)
        docker-compose ps
        ;;
    health)
        echo "Checking service health..."
        echo ""
        docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
        ;;
    clean)
        echo "Cleaning up containers and volumes..."
        read -p "This will remove all data. Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v
            echo "Cleanup complete."
        else
            echo "Cleanup cancelled."
        fi
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

echo ""