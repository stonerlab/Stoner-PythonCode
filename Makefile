PYTHON_SETUP	=	python setup.py

egg:
	$(PYTHON_SETUP) egg_info -d bdist_egg upload

doc: FORCE
	$(PYTHON_SETUP) build_sphinx upload_sphinx

FORCE:	
