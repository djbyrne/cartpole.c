// cartpole_module.c

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "cartpole.h"
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    CartPoleEnv env;
} PyCartPoleObject;

static int PyCartPole_init(PyCartPoleObject *self, PyObject *args, PyObject *kwds) {
    initialize(&self->env);
    return 0;
}

static void PyCartPole_dealloc(PyCartPoleObject *self) {
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PyCartPole_reset(PyCartPoleObject *self, PyObject *Py_UNUSED(ignored)) {
    reset(&self->env);
    // Return the initial state as a tuple
    return Py_BuildValue("(dddd)", self->env.x, self->env.x_dot, self->env.theta, self->env.theta_dot);
}

static PyObject *PyCartPole_step(PyCartPoleObject *self, PyObject *args) {
    int action;
    if (!PyArg_ParseTuple(args, "i", &action)) {
        return NULL;
    }

    double reward = 0.0;
    int done = step(&self->env, action, &reward);

    // Return a tuple: (state, reward, done)
    PyObject *state = Py_BuildValue("(dddd)", self->env.x, self->env.x_dot, self->env.theta, self->env.theta_dot);
    PyObject *result = Py_BuildValue("(OdO)", state, reward, PyBool_FromLong(done));
    Py_DECREF(state);
    return result;
}

static PyMethodDef PyCartPole_methods[] = {
    {"reset", (PyCFunction)PyCartPole_reset, METH_NOARGS, "Reset the environment to initial state"},
    {"step", (PyCFunction)PyCartPole_step, METH_VARARGS, "Take an action in the environment"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyCartPoleType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "cartpole.CartPoleEnv",
    .tp_basicsize = sizeof(PyCartPoleObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "CartPole Environment",
    .tp_methods = PyCartPole_methods,
    .tp_init = (initproc)PyCartPole_init,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor)PyCartPole_dealloc,
};

static PyModuleDef cartpolemodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "cartpole",
    .m_doc = "Python interface for the CartPole environment implemented in C",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_cartpole(void) {
    PyObject *m;
    if (PyType_Ready(&PyCartPoleType) < 0)
        return NULL;

    m = PyModule_Create(&cartpolemodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyCartPoleType);
    if (PyModule_AddObject(m, "CartPoleEnv", (PyObject *)&PyCartPoleType) < 0) {
        Py_DECREF(&PyCartPoleType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}