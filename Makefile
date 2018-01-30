ifeq ($(OS),"Windows_NT")
	SPHINX_BUILD	=	sphinx-build.bat
else
	SPHINX_BUILD	=	sphinx-build
endif
PYTHON_SETUP	=	python setup.py

clean:
	rm dist/*
	rm -rf build/*

test:
	$(PYTHON_SETUP) test

commit:
	$(MAKE) -C doc readme
	git commit -a
	git push origin master

wheel: clean test
	$(MAKE) -C doc readme
	$(PYTHON_SETUP) sdist bdist_wheel --universal
	twine upload dist/*

docbuild: FORCE
	$(MAKE) -C doc clean
	$(MAKE) -C doc html
	( cd ../gh-pages; git pull )
	rsync -rcm --perms --chmod=ugo=rwX --delete  --filter="P .git" --filter="P .nojekyll" doc/_build/html/ ../gh-pages/

FORCE:
