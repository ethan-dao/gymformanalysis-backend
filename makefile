# makefile - Development and deployment commands
.PHONY: install dev test build docker-build docker-run clean

# Installation
install:
	pip install -r requirements.txt

# Development
dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Testing
test:
	pytest test_api.py -v

test-load:
	locust -f locustfile.py --host=http://localhost:8000

# Docker
docker-build:
	docker build -t pullup-api:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# Kubernetes
k8s-deploy:
	kubectl apply -f kubernetes.yaml

k8s-delete:
	kubectl delete -f kubernetes.yaml

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	docker system prune -f

# Production deployment
deploy-prod:
	docker-compose -f docker-compose.prod.yml up -d

# Model training
train-model:
	python rnn_optimized.py

# Database migration (if using PostgreSQL instead of just Redis)
migrate:
	echo "Add database migration commands here"

# Monitoring
logs:
	docker-compose logs -f api

redis-cli:
	docker-compose exec redis redis-cli

# Health checks
health-check:
	curl -f http://localhost:8000/health || exit 1