/* _dot.c - a tiny C extension providing a dot product over NumPy buffers.
 *
 * It exists so that LinearRegression.predict can run its per-row dot
 * products through compiled C code when the extension is available.
 * The Python layer falls back to NumPy when it is not.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static PyObject *dot_product(PyObject *self, PyObject *args)
{
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return NULL;
    }
    PyArrayObject *a = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *b = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (a == NULL || b == NULL) {
        Py_XDECREF(a);
        Py_XDECREF(b);
        return NULL;
    }
    npy_intp n = PyArray_SIZE(a);
    if (PyArray_SIZE(b) != n) {
        Py_DECREF(a);
        Py_DECREF(b);
        PyErr_SetString(PyExc_ValueError, "vectors have different lengths");
        return NULL;
    }
    const double *av = (const double *)PyArray_DATA(a);
    const double *bv = (const double *)PyArray_DATA(b);
    double acc = 0.0;
    npy_intp i;
    for (i = 0; i < n; i++) {
        acc += av[i] * bv[i];
    }
    Py_DECREF(a);
    Py_DECREF(b);
    return PyFloat_FromDouble(acc);
}

static PyMethodDef dot_methods[] = {
    {"dot", dot_product, METH_VARARGS,
     "dot(a, b) -> float\n\nDot product of two 1D float64 arrays."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef dot_module = {
    PyModuleDef_HEAD_INIT, "_dot",
    "Compiled dot product used by NusaLearn linear models.", -1, dot_methods};

PyMODINIT_FUNC PyInit__dot(void)
{
    import_array();
    return PyModule_Create(&dot_module);
}
