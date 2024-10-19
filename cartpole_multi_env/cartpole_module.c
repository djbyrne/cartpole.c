// cartpole_module.c

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "cartpole.h"
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    CartPoleEnv env;
} PyCartPoleObject;

typedef struct {
    PyObject_HEAD
    CartPoleEnvBatch batch;
} PyCartPoleBatchObject;

static int PyCartPoleBatch_init(PyCartPoleBatchObject *self, PyObject *args, PyObject *kwds) {
    int num_envs = 1;
    static char *kwlist[] = {"num_envs", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &num_envs)) {
        return -1;
    }
    initialize_envs(&self->batch, num_envs);
    return 0;
}

static void PyCartPoleBatch_dealloc(PyCartPoleBatchObject *self) {
    free_envs(&self->batch);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject *PyCartPoleBatch_reset(PyCartPoleBatchObject *self, PyObject *args) {
    PyObject *reset_indices_obj = NULL;
    if (!PyArg_ParseTuple(args, "|O", &reset_indices_obj)) {
        return NULL;
    }

    int num_resets = self->batch.num_envs;
    int *reset_indices = NULL;

    if (reset_indices_obj && reset_indices_obj != Py_None) {
        if (!PyList_Check(reset_indices_obj)) {
            PyErr_SetString(PyExc_TypeError, "Expected a list of indices to reset");
            return NULL;
        }
        num_resets = (int)PyList_Size(reset_indices_obj);
        reset_indices = (int *)malloc(sizeof(int) * num_resets);
        for (int i = 0; i < num_resets; i++) {
            PyObject *item = PyList_GetItem(reset_indices_obj, i);
            if (!PyLong_Check(item)) {
                free(reset_indices);
                PyErr_SetString(PyExc_TypeError, "Indices must be integers");
                return NULL;
            }
            reset_indices[i] = (int)PyLong_AsLong(item);
        }
    } else {
        // Reset all environments if no indices provided
        num_resets = self->batch.num_envs;
        reset_indices = (int *)malloc(sizeof(int) * num_resets);
        for (int i = 0; i < num_resets; i++) {
            reset_indices[i] = i;
        }
    }

    // Call the reset function with the indices
    reset_envs(&self->batch, reset_indices, num_resets);

    free(reset_indices);

    // Build a list of states for the reset environments
    PyObject *state_list = PyList_New(num_resets);
    for (int i = 0; i < num_resets; i++) {
        int idx = reset_indices[i];
        CartPoleEnv *env = &self->batch.envs[idx];
        PyObject *state = Py_BuildValue("(dddd)", env->x, env->x_dot, env->theta, env->theta_dot);
        PyList_SET_ITEM(state_list, i, state);
    }

    return state_list;
}

static PyObject *PyCartPoleBatch_step(PyCartPoleBatchObject *self, PyObject *args) {
    PyObject *action_list;
    if (!PyArg_ParseTuple(args, "O", &action_list)) {
        return NULL;
    }

    if (!PyList_Check(action_list)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of actions");
        return NULL;
    }

    Py_ssize_t num_actions = PyList_Size(action_list);
    if (num_actions != self->batch.num_envs) {
        PyErr_SetString(PyExc_ValueError, "Number of actions must match number of environments");
        return NULL;
    }

    int *actions = (int *)malloc(sizeof(int) * self->batch.num_envs);
    double *rewards = (double *)malloc(sizeof(double) * self->batch.num_envs);
    int *dones = (int *)malloc(sizeof(int) * self->batch.num_envs);

    for (int i = 0; i < self->batch.num_envs; i++) {
        PyObject *item = PyList_GetItem(action_list, i);
        if (!PyLong_Check(item)) {
            free(actions); free(rewards); free(dones);
            PyErr_SetString(PyExc_TypeError, "Actions must be integers");
            return NULL;
        }
        actions[i] = (int)PyLong_AsLong(item);
    }

    // Step the environments
    step_envs(&self->batch, actions, rewards, dones);

    // Build the results
    PyObject *state_list = PyList_New(self->batch.num_envs);
    PyObject *reward_list = PyList_New(self->batch.num_envs);
    PyObject *done_list = PyList_New(self->batch.num_envs);

    for (int i = 0; i < self->batch.num_envs; i++) {
        CartPoleEnv *env = &self->batch.envs[i];
        PyObject *state = Py_BuildValue("(dddd)", env->x, env->x_dot, env->theta, env->theta_dot);
        PyList_SET_ITEM(state_list, i, state);
        PyList_SET_ITEM(reward_list, i, PyFloat_FromDouble(rewards[i]));
        PyList_SET_ITEM(done_list, i, PyBool_FromLong(dones[i]));
    }

    free(actions);
    free(rewards);
    free(dones);

    return Py_BuildValue("(OOO)", state_list, reward_list, done_list);
}

static PyMethodDef PyCartPoleBatch_methods[] = {
    {"reset", (PyCFunction)PyCartPoleBatch_reset, METH_VARARGS, "Reset environments by indices"},
    {"step", (PyCFunction)PyCartPoleBatch_step, METH_VARARGS, "Step all environments with a list of actions"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyCartPoleBatchType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "cartpole.CartPoleBatch",
    .tp_basicsize = sizeof(PyCartPoleBatchObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Batch of CartPole Environments",
    .tp_methods = PyCartPoleBatch_methods,
    .tp_init = (initproc)PyCartPoleBatch_init,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor)PyCartPoleBatch_dealloc,
};

static PyModuleDef cartpolemodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "cartpole",
    .m_doc = "Python interface for the CartPole environment implemented in C",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_cartpole(void) {
    PyObject *m;
    if (PyType_Ready(&PyCartPoleBatchType) < 0)
        return NULL;

    m = PyModule_Create(&cartpolemodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyCartPoleBatchType);
    if (PyModule_AddObject(m, "CartPoleBatch", (PyObject *)&PyCartPoleBatchType) < 0) {
        Py_DECREF(&PyCartPoleBatchType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}