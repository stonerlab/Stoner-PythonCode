ifeq ($(OS),"Windows_NT")
	SPHINX_BUILD	=	sphinx-build.bat
else
	SPHINX_BUILD	=	sphinx-build
endif
PYTHON_SETUP	=	python setup.py

egg:
	$(PYTHON_SETUP) egg_info -bb4 bdist_egg upload

wheel:
	$(PYTHON_SETUP) bdist_wheel --universal upload

doc: docbuild
	$(PYTHON_SETUP) upload_docs --upload-dir=doc/_build/html
	cp -ar doc/_build/html/* ../gh-pages/

docbuild: FORCE
	$(SPHINX_BUILD) -b html -a -E doc doc/_build/html

FORCE:
