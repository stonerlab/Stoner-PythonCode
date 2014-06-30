ifeq ($(OS),"Windows_NT")
	SPHINX_BUILD	=	sphinx-build.bat
else
	SPHINX_BUILD	=	sphinx-build
endif
PYTHON_SETUP	=	python setup.py

egg:
	$(PYTHON_SETUP) egg_info -d bdist_egg upload

doc: FORCE
	$(SPHINX_BUILD) -b html -a -E doc doc/_build/html
	$(PYTHON_SETUP) upload_sphinx

FORCE:	
