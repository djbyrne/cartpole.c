.Phony: build_c build_py
build_c:
	gcc -o cartpole main.c cartpole.c agent.c -lm

build_py:
	python setup.py build

clean:
	python setup.py clean

install:
	python setup.py install

build: build_c build_py install