test:
	python setup.py register -r pypitest; \
	python setup.py sdist upload -r pypitest

live:
	python setup.py register -r pypi; \
	python setup.py sdist upload -r pypi
