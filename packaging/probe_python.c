// Exercise relocated Python headers and a real extension import, without a GPU.
#include <Python.h>
static PyObject *answer(PyObject *self, PyObject *args) {
    return PyLong_FromLong(42);
}
static PyMethodDef methods[] = {{"answer", answer, METH_NOARGS, "Probe"}, {NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_hq_probe", NULL, -1, methods};
PyMODINIT_FUNC PyInit__hq_probe(void) { return PyModule_Create(&module); }
