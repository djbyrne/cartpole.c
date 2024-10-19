.Phony: build_c build_py clean install
build_c:
	gcc -o cartpole cartpole.c agent.c -lm

build_py:
	python setup.py build

clean:
	python setup.py clean

install:
	python setup.py install

build: build_c build_py