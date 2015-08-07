ifeq ($(OS),"Windows_NT")
	SPHINX_BUILD	=	sphinx-build.bat
else
	SPHINX_BUILD	=	sphinx-build
endif
PYTHON_SETUP	=	python setup.py

test:
	$(PYTHON_SETUP) test

egg: test
	$(PYTHON_SETUP) egg_info bdist_egg upload

wheel:
	$(PYTHON_SETUP) bdist_wheel --universal upload

doc: docbuild
	$(PYTHON_SETUP) upload_docs --upload-dir=doc/_build/html

docbuild: FORCE
	$(MAKE) -C doc clean
	$(MAKE) -C doc html
	rm -rfr doc/_build
	$(MAKE) -C doc html
	( cd ../gh-pages; git pull )
	rsync -rcm --perms --chmod=ugo=rwX --delete  --filter="P .git" --filter="P .nojekyll" doc/_build/html/ ../gh-pages/

FORCE:
