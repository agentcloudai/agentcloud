# Convenience shortcuts around docker compose. (Windows users without `make` can
# run the underlying `docker compose ...` commands shown in the README.)
.PHONY: up down logs shell download index ask eval test

up:            ## Build the app image and start the stack (DB + app)
	docker compose up -d --build

down:          ## Stop the stack (keeps data volumes)
	docker compose down

logs:          ## Follow logs
	docker compose logs -f

shell:         ## Open a shell in the app container
	docker compose exec app bash

download:      ## Download all AWS service guides (override: CSV=aws_all_doc_pdfs.csv)
	docker compose exec app rag-app download --csv $(or $(CSV),aws_service_guides.csv)

index:         ## Chunk + embed + store everything in data/
	docker compose exec app rag-app index

ask:           ## Ask a question:  make ask Q="How do I enable S3 versioning?"
	docker compose exec app rag-app ask "$(Q)"

eval:          ## Run the retrieval eval set
	docker compose exec app rag-app eval

test:          ## Run the test suite inside the container
	docker compose exec app python -m pytest -q
