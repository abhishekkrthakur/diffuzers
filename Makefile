.PHONY: quality style

quality:
	python -m black --check --line-length 119 --target-version py38 diffuzers/
	python -m isort --check-only diffuzers/
	python -m flake8 --max-line-length 119 diffuzers/

style:
	python -m black --line-length 119 --target-version py38 diffuzers/
	python -m isort diffuzers/