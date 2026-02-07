lint:
	ruff check .

format:
	black .

test:
	pytest -q

cov:
	pytest --cov=vecm_project --cov-report=term-missing --cov-fail-under=60

demo:
	python vecm_project/run_demo.py

signal:
	python -m vecm_project.scripts.daily_signal

notify:
	python -m vecm_project.scripts.notify --only-changed

docker-demo:
	docker compose up --build pipeline
