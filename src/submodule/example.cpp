#define NPY_NO_DEPRECATED_APINPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

static PyObject* test_sqrt(PyObject *self, PyObject *args)
{
    PyObject *arr_py;
    if (!PyArg_ParseTuple(args, "O", &arr_py))
        return PyErr_Format(PyExc_Exception, "Can't parse input");

    size_t size = PyArray_SIZE(arr_py);
    PyObject    *arr_tmp = PyArray_FROM_OTF(arr_py, NPY_FLOAT64, NPY_IN_ARRAY);  
    double *src = (double*)PyArray_DATA(arr_tmp);

    const npy_intp dims[1] = {(npy_intp)size};
    PyObject *ans = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double *dst = (double*)PyArray_DATA(ans);
    
    for (size_t i = 0; i < size; i++)
    {
        dst[i] = sqrt(src[i]);
    }
    return ans;
}


static PyMethodDef methods[] = {
		{
			"test_sqrt_c", test_sqrt, METH_VARARGS, "Example.",
		},
		{NULL, NULL, 0, NULL}
};


static struct PyModuleDef definition = {
		PyModuleDef_HEAD_INIT,
		"submodule_c_name",
		"Basic example",
		-1,
		methods
};


PyMODINIT_FUNC PyInit_submodule_c_name(void) {
	Py_Initialize();
	import_array();
	return PyModule_Create(&definition);
}
