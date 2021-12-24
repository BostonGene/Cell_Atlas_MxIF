#define NPY_NO_DEPRECATED_APINPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include <omp.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

void calculate_cell_contact_percentages(
		std::vector<std::vector<uint32_t>> &lst,
		std::vector<uint32_t> &cell_type,
		uint32_t *contact,
		int list_size,
		int n_classes,
		int threads)
{
	memset(contact, 0, n_classes * n_classes * sizeof(uint32_t));
	int n = lst.size();
	for (int i = 0; i < n; i++)
	{
		uint32_t tmp = cell_type[i] * n_classes;
		for (int j = 0; j < lst[i].size(); j++)
		{
			contact[tmp + cell_type[lst[i][j]]]++;
		}
	}
}

void sum_by_row(uint32_t *contact, uint32_t *sum_row, int n_classes)
{
	memset(sum_row, 0, n_classes * sizeof(uint32_t));
	for (int i = 0; i < n_classes; i++)
	{
		for (int j = 0; j < n_classes; j++)
			sum_row[i] += contact[i * n_classes + j];
	}
}

static PyObject* adjacency_label_permutation(PyObject *self, PyObject *args) {
	PyObject *arg1 = NULL;
	PyObject *arg2 = NULL;
	int32_t n_classes, permut, seed;
	int threads = 1;

	if (!PyArg_ParseTuple(args, "OOiiii", &arg1, &arg2, &n_classes, &permut, &threads, &seed))
		return PyErr_Format(PyExc_Exception, "Can't parse input adjacency_label_permutation.");

	if (threads < 0)
		threads = omp_get_max_threads();

	if (seed < 0)
		seed = std::chrono::steady_clock::now().time_since_epoch().count();

	// Init list of generators for each thread
	std::vector<std::mt19937> rng_lst(threads);
	for (int i = 0; i < threads; i++)
	{
		std::mt19937 rng((unsigned int)(seed + i));
		rng_lst[i] = rng;
	}

	int n_classes2 = n_classes * n_classes;
	// Get size of list
	int list_size = (int) PyList_Size(arg1);

	// Get cell types
	uint32_t *tmp;
	tmp = (uint32_t *) PyArray_DATA(PyArray_FROM_OTF(arg2, NPY_UINT32, NPY_IN_ARRAY));
	std::vector<uint32_t> cell_type(tmp, tmp + list_size);

	// Get adjacency matrix
	std::vector<std::vector<uint32_t>> lst(list_size);
	for (int i = 0; i < list_size; i++)
	{
		PyObject *py_arr = PyArray_FROM_OTF(PyList_GetItem(arg1, (Py_ssize_t) i), NPY_UINT32, NPY_IN_ARRAY);
		int n = (int) PyArray_DIM(py_arr, 0);
		tmp = (uint32_t *)PyArray_DATA(py_arr);
		std::vector<uint32_t> v(tmp, tmp + n);
		lst[i] = v;
		Py_DECREF(py_arr);
	}

	// Init memory for multi-threading
	std::vector<std::vector<uint32_t>> cell_type_lst(threads);
	for (int i = 0; i < threads; i++)
		cell_type_lst[i] = cell_type;

	int malloc_error = 0;
	uint32_t **contact_lst = (uint32_t**)calloc(threads, sizeof(uint32_t*));
	if (contact_lst)
	{
		for (int i = 0; i < threads; i++)
		{
			contact_lst[i] = (uint32_t *)malloc(n_classes2 * sizeof(uint32_t));
		}
	}
	else
		malloc_error = 1;

	uint32_t **sum_row_lst = (uint32_t**)calloc(threads, sizeof(uint32_t*));
	if (sum_row_lst)
	{
		for (int i = 0; i < threads; i++)
			sum_row_lst[i] = (uint32_t*)malloc(n_classes * sizeof(uint32_t));
	}
	else
		malloc_error = 1;

	// Init memory for saving results
	double *ref_cell_contact = (double*)malloc(n_classes2 * sizeof(double));
	uint32_t *contact_likelihood = (uint32_t*)malloc(n_classes2 * sizeof(uint32_t));
	double *contact_likelihood_result = (double*)malloc(n_classes2 * sizeof(double));

	if (malloc_error)
	{
		free(contact_likelihood_result);
		free(ref_cell_contact);
		free(contact_likelihood);
		if (contact_lst)
			for (int i = 0; i < threads; i++)
				free(contact_lst[i]);

		if (sum_row_lst)
			for (int i = 0; i < threads; i++)
				free(sum_row_lst[i]);
		free(contact_lst);
		free(sum_row_lst);
		return PyErr_Format(PyExc_Exception, "Can't allocate memory");
	}

	// Init initial values
	for (int i = 0; i < n_classes2; i++)
		contact_likelihood[i] = 1;

	// Get reference matrix
	calculate_cell_contact_percentages(lst, cell_type, contact_lst[0], list_size, n_classes, threads);
	sum_by_row(contact_lst[0], sum_row_lst[0], n_classes);

	for (int i = 0; i < n_classes2; i++)
		ref_cell_contact[i] = (double)contact_lst[0][i] / (double)sum_row_lst[0][i % n_classes];

	// Calculate likelihood
	#pragma omp parallel num_threads(threads)
	{
		#pragma omp for
		for (int i = 0; i < permut; i++)
		{
			int tid = omp_get_thread_num();
			shuffle(cell_type_lst[tid].begin(), cell_type_lst[tid].end(), rng_lst[tid]);
			calculate_cell_contact_percentages(lst, cell_type_lst[tid], contact_lst[tid], list_size, n_classes, threads);
			sum_by_row(contact_lst[tid], sum_row_lst[tid], n_classes);
			for (int j = 0; j < n_classes2; j++)
			{
				// Need to wait for access to memory
				#pragma omp atomic
				contact_likelihood[j] += ((double)contact_lst[tid][j] / (double)sum_row_lst[tid][j % n_classes]) >= ref_cell_contact[j];
			}
		}
	}

	for (int i = 0; i < n_classes2; i++)
		contact_likelihood_result[i] = (double)contact_likelihood[i] / (permut + 1);

	// Write the answer
	npy_intp dims[2] = {n_classes, n_classes};
	PyObject *ans = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
	memcpy((double*)PyArray_DATA(ans), contact_likelihood_result, n_classes2 * sizeof(double));

	// Free memory
	free(contact_likelihood_result);
	free(ref_cell_contact);
	free(contact_likelihood);
	for (int i = 0; i < threads; i++)
		free(contact_lst[i]);
	for (int i = 0; i < threads; i++)
		free(sum_row_lst[i]);
	free(contact_lst);
	free(sum_row_lst);
	return (ans);
}

static PyMethodDef methods[] = {
		{
				"adjacency_label_permutation_c",
				adjacency_label_permutation, METH_VARARGS,
				"Calculate adjacency permutation by cell types.",
		},
		{NULL, NULL, 0, NULL}
};

static struct PyModuleDef definition = {
		PyModuleDef_HEAD_INIT,
		"permutation_c",
		"C api functions for permutation module",
		-1,
		methods
};

PyMODINIT_FUNC PyInit_permutation_c(void) {
	Py_Initialize();
	import_array();
	return PyModule_Create(&definition);
}