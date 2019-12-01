ifeq ($(OS),"Windows_NT")
	SPHINX_BUILD	=	sphinx-build.bat
else
	SPHINX_BUILD	=	sphinx-build
endif
PYTHON_SETUP	=	python setup.py

clean:
	$(MAKE) -C doc clean
	rm dist/*
	rm -rf build/*
	find -name '__pycache__' -exec rm -rf {} \;

test:
	$(PYTHON_SETUP) test

check:
	prospector -E -0 --profile-path=. -P .landscape.yml Stoner > prospector-report.txt

black:
	find Stoner -name '*.py' -exec black -l 119 {} \;
	find doc/samples -name '*.py' -exec black {} \;
	find scripts -name '*.py' -exec black {} \;

commit: black
	$(MAKE) -C doc readme
	git commit -a
	git push origin master

_build_wheel:
	$(MAKE) -C doc readme
	$(PYTHON_SETUP) sdist bdist_wheel --universal
	twine upload dist/*

wheel: clean test _build_wheel

docbuild: FORCE
	$(MAKE) -C doc clean
	$(MAKE) -C doc html
	( cd ../gh-pages; git pull )
	rsync -rcm --perms --chmod=ugo=rwX --delete  --filter="P .git" --filter="P .nojekyll" doc/_build/html/ ../gh-pages/

rtdbuild: export READTHEDOCS=1
rtdbuild: docbuild

FORCE:
