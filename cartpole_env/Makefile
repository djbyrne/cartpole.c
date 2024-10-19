.Phony: build_c build_py
build_c:
	gcc -o cartpole main.c cartpole.c agent.c -lm

build_py:
	python setup.py build

install:
	python setup.py install