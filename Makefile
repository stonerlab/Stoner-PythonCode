ifeq ($(OS),"Windows_NT")
	SPHINX_BUILD	=	sphinx-build.bat
else
	SPHINX_BUILD	=	sphinx-build
endif
PYTHON_SETUP	=	python setup.py

egg:
	$(PYTHON_SETUP) egg_info -bb2 bdist_egg upload

doc: docbuild
	$(PYTHON_SETUP) upload_sphinx
	cp -ar doc/_build/html/* ../gh-pages/

docbuild: FORCE
	$(SPHINX_BUILD) -b html -a -E doc doc/_build/html

FORCE:
