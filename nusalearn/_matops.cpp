// _matops.cpp - a small C++ extension providing a dense matrix multiply.
//
// LinearRegression's normal equation path needs X^T X and X^T y. When
// this extension is compiled, the X^T X product is computed here; the
// Python layer falls back to NumPy otherwise. The implementation is a
// plain triple loop written for this project.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace {

PyObject *matmul(PyObject * /* self */, PyObject *args)
{
    PyObject *a_obj = nullptr;
    PyObject *b_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return nullptr;
    }
    PyArrayObject *a = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject *b = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (a == nullptr || b == nullptr) {
        Py_XDECREF(a);
        Py_XDECREF(b);
        return nullptr;
    }
    if (PyArray_NDIM(a) != 2 || PyArray_NDIM(b) != 2) {
        Py_DECREF(a);
        Py_DECREF(b);
        PyErr_SetString(PyExc_ValueError, "matmul expects 2D arrays");
        return nullptr;
    }
    const npy_intp rows = PyArray_DIM(a, 0);
    const npy_intp inner = PyArray_DIM(a, 1);
    const npy_intp cols = PyArray_DIM(b, 1);
    if (PyArray_DIM(b, 0) != inner) {
        Py_DECREF(a);
        Py_DECREF(b);
        PyErr_SetString(PyExc_ValueError, "inner dimensions do not match");
        return nullptr;
    }
    npy_intp dims[2] = {rows, cols};
    PyArrayObject *out = reinterpret_cast<PyArrayObject *>(
        PyArray_SimpleNew(2, dims, NPY_DOUBLE));
    if (out == nullptr) {
        Py_DECREF(a);
        Py_DECREF(b);
        return nullptr;
    }
    const double *av = static_cast<const double *>(PyArray_DATA(a));
    const double *bv = static_cast<const double *>(PyArray_DATA(b));
    double *ov = static_cast<double *>(PyArray_DATA(out));
    for (npy_intp i = 0; i < rows; i++) {
        for (npy_intp j = 0; j < cols; j++) {
            double acc = 0.0;
            for (npy_intp k = 0; k < inner; k++) {
                acc += av[i * inner + k] * bv[k * cols + j];
            }
            ov[i * cols + j] = acc;
        }
    }
    Py_DECREF(a);
    Py_DECREF(b);
    return reinterpret_cast<PyObject *>(out);
}

PyMethodDef matops_methods[] = {
    {"matmul", matmul, METH_VARARGS,
     "matmul(A, B) -> ndarray\n\nDense matrix product of two 2D arrays."},
    {nullptr, nullptr, 0, nullptr}};

struct PyModuleDef matops_module = {
    PyModuleDef_HEAD_INIT, "_matops",
    "Compiled matrix multiply used by NusaLearn linear models.", -1,
    matops_methods};

}  // namespace

PyMODINIT_FUNC PyInit__matops(void)
{
    import_array();
    return PyModule_Create(&matops_module);
}
