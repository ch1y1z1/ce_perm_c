/* -*-c-*-  */
/*
 * _superlu module
 *
 * Python interface to SuperLU decompositions.
 */

/* Copyright 1999 Travis Oliphant
 *
 * Permission to copy and modified this file is granted under
 * the revised BSD license. No warranty is expressed or IMPLIED
 */

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL _scipy_sparse_superlu_ARRAY_API
#include <numpy/ndarrayobject.h>

#include "_superluobject.h"


/*
 * NULL-safe deconstruction functions
 */
void XDestroy_SuperMatrix_Store(SuperMatrix * A)
{
    Destroy_SuperMatrix_Store(A);	/* safe as-is */
    A->Store = NULL;
}

void XDestroy_SuperNode_Matrix(SuperMatrix * A)
{
    if (A->Store) {
	Destroy_SuperNode_Matrix(A);
    }
    A->Store = NULL;
}

void XDestroy_CompCol_Matrix(SuperMatrix * A)
{
    if (A->Store) {
	Destroy_CompCol_Matrix(A);
    }
    A->Store = NULL;
}

void XDestroy_CompCol_Permuted(SuperMatrix * A)
{
    if (A->Store) {
	Destroy_CompCol_Permuted(A);
    }
    A->Store = NULL;
}

void XStatFree(SuperLUStat_t * stat)
{
    if (stat->ops) {
	StatFree(stat);
    }
    stat->ops = NULL;
}

void
zgssv(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
      SuperMatrix *L, SuperMatrix *U, SuperMatrix *B,
      SuperLUStat_t *stat, int *info )
{

	DNformat *Bstore;
	SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
	SuperMatrix AC; /* Matrix postmultiplied by Pc */
	int      lwork = 0, *etree, i;
	GlobalLU_t Glu; /* Not needed on return. */

	/* Set default values for some parameters */
	int      panel_size;     /* panel size */
	int      relax;          /* no of columns in a relaxed snodes */
	int      permc_spec;
	trans_t  trans = NOTRANS;
	double   *utime;
	double   t;	/* Temporary time */

	/* Test the input parameters ... */
	*info = 0;
	Bstore = B->Store;
	if ( options->Fact != DOFACT ) *info = -1;
	else if ( A->nrow != A->ncol || A->nrow < 0 ||
	          (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
	          A->Dtype != SLU_Z || A->Mtype != SLU_GE )
		*info = -2;
	else if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
	          B->Stype != SLU_DN || B->Dtype != SLU_Z || B->Mtype != SLU_GE )
		*info = -7;
	if ( *info != 0 ) {
		i = -(*info);
		input_error("zgssv", &i);
		return;
	}

	utime = stat->utime;

	/* Convert A to SLU_NC format when necessary. */
	if ( A->Stype == SLU_NR ) {
		NRformat *Astore = A->Store;
		AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
		zCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz,
		                       Astore->nzval, Astore->colind, Astore->rowptr,
		                       SLU_NC, A->Dtype, A->Mtype);
		trans = TRANS;
	} else {
		if ( A->Stype == SLU_NC ) AA = A;
	}

	t = SuperLU_timer_();
	/*
	 * Get column permutation vector perm_c[], according to permc_spec:
	 *   permc_spec = NATURAL:  natural ordering
	 *   permc_spec = MMD_AT_PLUS_A: minimum degree on structure of A'+A
	 *   permc_spec = MMD_ATA:  minimum degree on structure of A'*A
	 *   permc_spec = COLAMD:   approximate minimum degree column ordering
	 *   permc_spec = MY_PERMC: the ordering already supplied in perm_c[]
	 */
	permc_spec = options->ColPerm;
	if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
		get_perm_c(permc_spec, AA, perm_c);
	utime[COLPERM] = SuperLU_timer_() - t;

	etree = intMalloc(A->ncol);

	t = SuperLU_timer_();
	sp_preorder(options, AA, perm_c, etree, &AC);
	utime[ETREE] = SuperLU_timer_() - t;

	panel_size = sp_ienv(1);
	relax = sp_ienv(2);

	/*printf("Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n",
	  relax, panel_size, sp_ienv(3), sp_ienv(4));*/
	t = SuperLU_timer_();
	/* Compute the LU factorization of A. */
	zgstrf(options, &AC, relax, panel_size, etree,
	       NULL, lwork, perm_c, perm_r, L, U, &Glu, stat, info);
	utime[FACT] = SuperLU_timer_() - t;

	t = SuperLU_timer_();
	if ( *info == 0 ) {
		/* Solve the system A*X=B, overwriting B with X. */
		zgstrs (trans, L, U, perm_c, perm_r, B, stat, info);
	}
	utime[SOLVE] = SuperLU_timer_() - t;

	SUPERLU_FREE (etree);
	Destroy_CompCol_Permuted(&AC);
	if ( A->Stype == SLU_NR ) {
		Destroy_SuperMatrix_Store(AA);
		SUPERLU_FREE(AA);
	}

}



/*
 * Data-type dependent implementations for Xgssv and Xgstrf;
 *
 * These have to included from separate files because of SuperLU include
 * structure.
 */

static PyObject *Py_gssv(PyObject * self, PyObject * args,
			 PyObject * kwdict)
{
    volatile PyObject *Py_B = NULL;
    volatile PyArrayObject *Py_X = NULL;
    volatile PyArrayObject *nzvals = NULL;
    volatile PyArrayObject *colind = NULL, *rowptr = NULL;
    volatile int N, nnz;
    volatile int info;
    volatile int csc = 0;
    volatile int *perm_r = NULL, *perm_c = NULL;
    volatile SuperMatrix A = { 0 }, B = { 0 }, L = { 0 }, U = { 0 };
    volatile superlu_options_t options = { 0 };
    volatile SuperLUStat_t stat = { 0 };
    volatile PyObject *option_dict = NULL;
    volatile int type;
    volatile jmp_buf *jmpbuf_ptr;
    SLU_BEGIN_THREADS_DEF;

    static char *kwlist[] = {
        "N", "nnz", "nzvals", "colind", "rowptr", "B", "csc",
        "options", NULL
    };

    /* Get input arguments */
    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "iiO!O!O!O|iO", kwlist,
				     &N, &nnz, &PyArray_Type, &nzvals,
				     &PyArray_Type, &colind, &PyArray_Type,
				     &rowptr, &Py_B, &csc, &option_dict)) {
	return NULL;
    }

    if (!_CHECK_INTEGER(colind) || !_CHECK_INTEGER(rowptr)) {
	PyErr_SetString(PyExc_TypeError,
			"colind and rowptr must be of type cint");
	return NULL;
    }

    type = PyArray_TYPE((PyArrayObject*)nzvals);
    if (!CHECK_SLU_TYPE(type)) {
	PyErr_SetString(PyExc_TypeError,
			"nzvals is not of a type supported by SuperLU");
	return NULL;
    }

    if (!set_superlu_options_from_dict((superlu_options_t*)&options, 0,
                                       (PyObject*)option_dict, NULL, NULL)) {
	return NULL;
    }

    /* Create Space for output */
    Py_X = (PyArrayObject*)PyArray_FROMANY(
        (PyObject*)Py_B, type, 1, 2,
        NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
    if (Py_X == NULL)
	return NULL;

    if (PyArray_DIM((PyArrayObject*)Py_X, 0) != N) {
        PyErr_SetString(PyExc_ValueError,
                        "b array has invalid shape");
        Py_DECREF(Py_X);
        return NULL;
    }

    if (csc) {
	if (NCFormat_from_spMatrix((SuperMatrix*)&A, N, N, nnz,
                                   (PyArrayObject *)nzvals, (PyArrayObject *)colind,
                                   (PyArrayObject *)rowptr, type)) {
	    Py_DECREF(Py_X);
	    return NULL;
	}
    }
    else {
	if (NRFormat_from_spMatrix((SuperMatrix*)&A, N, N, nnz, (PyArrayObject *)nzvals,
                                   (PyArrayObject *)colind, (PyArrayObject *)rowptr,
				   type)) {
	    Py_DECREF(Py_X);
	    return NULL;
	}
    }

    if (DenseSuper_from_Numeric((SuperMatrix*)&B, (PyObject*)Py_X)) {
	Destroy_SuperMatrix_Store((SuperMatrix*)&A);
	Py_DECREF(Py_X);
	return NULL;
    }

    /* B and Py_X  share same data now but Py_X "owns" it */

    /* Setup options */

    jmpbuf_ptr = (volatile jmp_buf *)superlu_python_jmpbuf();
    SLU_BEGIN_THREADS;
    if (setjmp(*(jmp_buf*)jmpbuf_ptr)) {
        SLU_END_THREADS;
	goto fail;
    }
    else {
	perm_c = intMalloc(N);
	perm_r = intMalloc(N);
	StatInit((SuperLUStat_t*)&stat);

	/* Compute direct inverse of sparse Matrix */
	gssv(type, (superlu_options_t*)&options, (SuperMatrix*)&A, (int*)perm_c, (int*)perm_r,
             (SuperMatrix*)&L, (SuperMatrix*)&U, (SuperMatrix*)&B, (SuperLUStat_t*)&stat,
             (int*)&info);
        SLU_END_THREADS;
    }

    SUPERLU_FREE((void*)perm_r);
    SUPERLU_FREE((void*)perm_c);
    Destroy_SuperMatrix_Store((SuperMatrix*)&A);	/* holds just a pointer to the data */
    Destroy_SuperMatrix_Store((SuperMatrix*)&B);
    Destroy_SuperNode_Matrix((SuperMatrix*)&L);
    Destroy_CompCol_Matrix((SuperMatrix*)&U);
    StatFree((SuperLUStat_t*)&stat);

    return Py_BuildValue("Ni", Py_X, info);

  fail:
    SUPERLU_FREE((void*)perm_r);
    SUPERLU_FREE((void*)perm_c);
    XDestroy_SuperMatrix_Store((SuperMatrix*)&A);	/* holds just a pointer to the data */
    XDestroy_SuperMatrix_Store((SuperMatrix*)&B);
    XDestroy_SuperNode_Matrix((SuperMatrix*)&L);
    XDestroy_CompCol_Matrix((SuperMatrix*)&U);
    XStatFree((SuperLUStat_t*)&stat);
    Py_XDECREF(Py_X);
    return NULL;
}

#include "SuperLU/SRC/slu_zdefs.h"

// hack here
void
zgssv_1(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
      SuperMatrix *L, SuperMatrix *U, SuperMatrix *B,
      SuperLUStat_t *stat, int *info )
{

	DNformat *Bstore;
	SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
	SuperMatrix AC; /* Matrix postmultiplied by Pc */
	int      lwork = 0, *etree, i;
	GlobalLU_t Glu; /* Not needed on return. */

	/* Set default values for some parameters */
	int      panel_size;     /* panel size */
	int      relax;          /* no of columns in a relaxed snodes */
	int      permc_spec;
	trans_t  trans = NOTRANS;
	double   *utime;
	double   t;	/* Temporary time */

	/* Test the input parameters ... */
	*info = 0;
	Bstore = B->Store;
	if ( options->Fact != DOFACT ) *info = -1;
	else if ( A->nrow != A->ncol || A->nrow < 0 ||
	          (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
	          A->Dtype != SLU_Z || A->Mtype != SLU_GE )
		*info = -2;
	else if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
	          B->Stype != SLU_DN || B->Dtype != SLU_Z || B->Mtype != SLU_GE )
		*info = -7;
	if ( *info != 0 ) {
		i = -(*info);
		input_error("zgssv", &i);
		return;
	}

	utime = stat->utime;

	/* Convert A to SLU_NC format when necessary. */
	if ( A->Stype == SLU_NR ) {
		NRformat *Astore = A->Store;
		AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
		zCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz,
		                       Astore->nzval, Astore->colind, Astore->rowptr,
		                       SLU_NC, A->Dtype, A->Mtype);
		trans = TRANS;
	} else {
		if ( A->Stype == SLU_NC ) AA = A;
	}

	t = SuperLU_timer_();
	/*
	 * Get column permutation vector perm_c[], according to permc_spec:
	 *   permc_spec = NATURAL:  natural ordering
	 *   permc_spec = MMD_AT_PLUS_A: minimum degree on structure of A'+A
	 *   permc_spec = MMD_ATA:  minimum degree on structure of A'*A
	 *   permc_spec = COLAMD:   approximate minimum degree column ordering
	 *   permc_spec = MY_PERMC: the ordering already supplied in perm_c[]
	 */
	permc_spec = options->ColPerm;
//	printf("calling get_perm_c\n");
	if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
		get_perm_c(permc_spec, AA, perm_c);
	utime[COLPERM] = SuperLU_timer_() - t;

//	Destroy_CompCol_Permuted(&AC);
	if ( A->Stype == SLU_NR ) {
		Destroy_SuperMatrix_Store(AA);
		SUPERLU_FREE(AA);
	}
}

void dest_perm_c(PyObject *py_perm_c)
{
//	printf("calling gc\n");
	void *perm_c = PyCapsule_GetPointer(py_perm_c, "perm_c");
	SUPERLU_FREE((int*)perm_c);
}

static PyObject *Py_zgssv_1(PyObject * self, PyObject * args,
                            PyObject * kwdict)
{
	volatile PyObject *Py_B = NULL;
	volatile PyArrayObject *Py_X = NULL;
	volatile PyArrayObject *nzvals = NULL;
	volatile PyArrayObject *colind = NULL, *rowptr = NULL;
	volatile int N, nnz;
	volatile int info;
	volatile int csc = 0;
	volatile int *perm_r = NULL, *perm_c = NULL;
	volatile SuperMatrix A = { 0 }, B = { 0 }, L = { 0 }, U = { 0 };
	volatile superlu_options_t options = { 0 };
	volatile SuperLUStat_t stat = { 0 };
	volatile PyObject *option_dict = NULL;
	volatile int type;
	volatile jmp_buf *jmpbuf_ptr;
	SLU_BEGIN_THREADS_DEF;

	static char *kwlist[] = {
			"N", "nnz", "nzvals", "colind", "rowptr", "B", "csc",
			"options", NULL
	};

	/* Get input arguments */
	if (!PyArg_ParseTupleAndKeywords(args, kwdict, "iiO!O!O!O|iO", kwlist,
	                                 &N, &nnz, &PyArray_Type, &nzvals,
	                                 &PyArray_Type, &colind, &PyArray_Type,
	                                 &rowptr, &Py_B, &csc, &option_dict)) {
		return NULL;
	}

	if (!_CHECK_INTEGER(colind) || !_CHECK_INTEGER(rowptr)) {
		PyErr_SetString(PyExc_TypeError,
		                "colind and rowptr must be of type cint");
		return NULL;
	}

	type = PyArray_TYPE((PyArrayObject*)nzvals);
	if (!CHECK_SLU_TYPE(type)) {
		PyErr_SetString(PyExc_TypeError,
		                "nzvals is not of a type supported by SuperLU");
		return NULL;
	}

	if (!set_superlu_options_from_dict((superlu_options_t*)&options, 0,
	                                   (PyObject*)option_dict, NULL, NULL)) {
		return NULL;
	}

	/* Create Space for output */
	Py_X = (PyArrayObject*)PyArray_FROMANY(
			(PyObject*)Py_B, type, 1, 2,
			NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
	if (Py_X == NULL)
		return NULL;

	if (PyArray_DIM((PyArrayObject*)Py_X, 0) != N) {
		PyErr_SetString(PyExc_ValueError,
		                "b array has invalid shape");
		Py_DECREF(Py_X);
		return NULL;
	}

	if (csc) {
		if (NCFormat_from_spMatrix((SuperMatrix*)&A, N, N, nnz,
		                           (PyArrayObject *)nzvals, (PyArrayObject *)colind,
		                           (PyArrayObject *)rowptr, type)) {
			Py_DECREF(Py_X);
			return NULL;
		}
	}
	else {
		if (NRFormat_from_spMatrix((SuperMatrix*)&A, N, N, nnz, (PyArrayObject *)nzvals,
		                           (PyArrayObject *)colind, (PyArrayObject *)rowptr,
		                           type)) {
			Py_DECREF(Py_X);
			return NULL;
		}
	}

	if (DenseSuper_from_Numeric((SuperMatrix*)&B, (PyObject*)Py_X)) {
		Destroy_SuperMatrix_Store((SuperMatrix*)&A);
		Py_DECREF(Py_X);
		return NULL;
	}

	/* B and Py_X  share same data now but Py_X "owns" it */

	/* Setup options */

	jmpbuf_ptr = (volatile jmp_buf *)superlu_python_jmpbuf();
	SLU_BEGIN_THREADS;
	if (setjmp(*(jmp_buf*)jmpbuf_ptr)) {
		SLU_END_THREADS;
		goto fail;
	}
	else {
		perm_c = intMalloc(N);
		perm_r = intMalloc(N);
		StatInit((SuperLUStat_t *) &stat);

		/* Compute direct inverse of sparse Matrix */
		zgssv_1((superlu_options_t*)&options, (SuperMatrix*)&A, (int*)perm_c, (int*)perm_r,
		     (SuperMatrix*)&L, (SuperMatrix*)&U, (SuperMatrix*)&B, (SuperLUStat_t*)&stat,
		     (int*)&info);
		SLU_END_THREADS;
	}
//	printf("zgssv return\n");

	SUPERLU_FREE((void*)perm_r);
//	SUPERLU_FREE((void*)perm_c);
	Destroy_SuperMatrix_Store((SuperMatrix*)&A);	/* holds just a pointer to the data */
	Destroy_SuperMatrix_Store((SuperMatrix*)&B);
//	Destroy_SuperNode_Matrix((SuperMatrix*)&L);
//	Destroy_CompCol_Matrix((SuperMatrix*)&U);
	StatFree((SuperLUStat_t*)&stat);

	// 泄漏：
	Py_XDECREF(Py_X);

	return PyCapsule_New((void*)perm_c, "perm_c", dest_perm_c);

	fail:
	SUPERLU_FREE((void*)perm_r);
	SUPERLU_FREE((void*)perm_c);
	XDestroy_SuperMatrix_Store((SuperMatrix*)&A);	/* holds just a pointer to the data */
	XDestroy_SuperMatrix_Store((SuperMatrix*)&B);
	XDestroy_SuperNode_Matrix((SuperMatrix*)&L);
	XDestroy_CompCol_Matrix((SuperMatrix*)&U);
	XStatFree((SuperLUStat_t*)&stat);
	Py_XDECREF(Py_X);
	return NULL;
}

void
zgssv_2(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
        SuperMatrix *L, SuperMatrix *U, SuperMatrix *B,
        SuperLUStat_t *stat, int *info )
{

	DNformat *Bstore;
	SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
	SuperMatrix AC; /* Matrix postmultiplied by Pc */
	int      lwork = 0, *etree, i;
	GlobalLU_t Glu; /* Not needed on return. */

	/* Set default values for some parameters */
	int      panel_size;     /* panel size */
	int      relax;          /* no of columns in a relaxed snodes */
	int      permc_spec;
	trans_t  trans = NOTRANS;
	double   *utime;
	double   t;	/* Temporary time */

	/* Test the input parameters ... */
	*info = 0;
	Bstore = B->Store;
	if ( options->Fact != DOFACT ) *info = -1;
	else if ( A->nrow != A->ncol || A->nrow < 0 ||
	          (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
	          A->Dtype != SLU_Z || A->Mtype != SLU_GE )
		*info = -2;
	else if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
	          B->Stype != SLU_DN || B->Dtype != SLU_Z || B->Mtype != SLU_GE )
		*info = -7;
	if ( *info != 0 ) {
		i = -(*info);
		input_error("zgssv", &i);
		return;
	}

	utime = stat->utime;

	/* Convert A to SLU_NC format when necessary. */
	if ( A->Stype == SLU_NR ) {
		NRformat *Astore = A->Store;
		AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
		zCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz,
		                       Astore->nzval, Astore->colind, Astore->rowptr,
		                       SLU_NC, A->Dtype, A->Mtype);
		trans = TRANS;
	} else {
		if ( A->Stype == SLU_NC ) AA = A;
	}

	t = SuperLU_timer_();
	/*
	 * Get column permutation vector perm_c[], according to permc_spec:
	 *   permc_spec = NATURAL:  natural ordering
	 *   permc_spec = MMD_AT_PLUS_A: minimum degree on structure of A'+A
	 *   permc_spec = MMD_ATA:  minimum degree on structure of A'*A
	 *   permc_spec = COLAMD:   approximate minimum degree column ordering
	 *   permc_spec = MY_PERMC: the ordering already supplied in perm_c[]
	 */
	permc_spec = options->ColPerm;
//	printf("calling get_perm_c\n");
//	if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
//		get_perm_c(permc_spec, AA, perm_c);
	utime[COLPERM] = SuperLU_timer_() - t;

	etree = intMalloc(A->ncol);

	t = SuperLU_timer_();
//	printf("calling sp_preorder\n");
	sp_preorder(options, AA, perm_c, etree, &AC);
	utime[ETREE] = SuperLU_timer_() - t;

	panel_size = sp_ienv(1);
	relax = sp_ienv(2);

	/*printf("Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n",
	  relax, panel_size, sp_ienv(3), sp_ienv(4));*/
	t = SuperLU_timer_();
	/* Compute the LU factorization of A. */
//	printf("calling zgstrf\n");
	zgstrf(options, &AC, relax, panel_size, etree,
	       NULL, lwork, perm_c, perm_r, L, U, &Glu, stat, info);
	utime[FACT] = SuperLU_timer_() - t;

	t = SuperLU_timer_();
	if ( *info == 0 ) {
		/* Solve the system A*X=B, overwriting B with X. */
//		printf("calling zgstrs\n");
		zgstrs (trans, L, U, perm_c, perm_r, B, stat, info);
	}
	utime[SOLVE] = SuperLU_timer_() - t;

	SUPERLU_FREE (etree);
	Destroy_CompCol_Permuted(&AC);
	if ( A->Stype == SLU_NR ) {
		Destroy_SuperMatrix_Store(AA);
		SUPERLU_FREE(AA);
	}
}

static PyObject *Py_zgssv_2(PyObject * self, PyObject * args,
                            PyObject * kwdict)
{
	volatile PyObject *Py_B = NULL;
	volatile PyArrayObject *Py_X = NULL;
	volatile PyArrayObject *nzvals = NULL;
	volatile PyArrayObject *colind = NULL, *rowptr = NULL;
	volatile int N, nnz;
	volatile int info;
	volatile int csc = 0;
	volatile int *perm_r = NULL, *perm_c = NULL;
	volatile SuperMatrix A = { 0 }, B = { 0 }, L = { 0 }, U = { 0 };
	volatile superlu_options_t options = { 0 };
	volatile SuperLUStat_t stat = { 0 };
	volatile PyObject *option_dict = NULL;
	volatile int type;
	volatile jmp_buf *jmpbuf_ptr;
	volatile PyObject *py_perm_c = NULL;
	SLU_BEGIN_THREADS_DEF;

	static char *kwlist[] = {
			"N", "nnz", "nzvals", "colind", "rowptr", "B", "csc",
			"options", "perm_c", NULL
	};

	/* Get input arguments */
	if (!PyArg_ParseTupleAndKeywords(args, kwdict, "iiO!O!O!O|iOO", kwlist,
	                                 &N, &nnz, &PyArray_Type, &nzvals,
	                                 &PyArray_Type, &colind, &PyArray_Type,
	                                 &rowptr, &Py_B, &csc, &option_dict, &py_perm_c)) {
		return NULL;
	}

	perm_c = PyCapsule_GetPointer(py_perm_c, "perm_c");
	if (perm_c == NULL) {
		PyErr_SetString(PyExc_TypeError,
		                "perm_c must be a valid capsule");
		return NULL;
	}

	if (!_CHECK_INTEGER(colind) || !_CHECK_INTEGER(rowptr)) {
		PyErr_SetString(PyExc_TypeError,
		                "colind and rowptr must be of type cint");
		return NULL;
	}

	type = PyArray_TYPE((PyArrayObject*)nzvals);
	if (!CHECK_SLU_TYPE(type)) {
		PyErr_SetString(PyExc_TypeError,
		                "nzvals is not of a type supported by SuperLU");
		return NULL;
	}

	if (!set_superlu_options_from_dict((superlu_options_t*)&options, 0,
	                                   (PyObject*)option_dict, NULL, NULL)) {
		return NULL;
	}

	/* Create Space for output */
	Py_X = (PyArrayObject*)PyArray_FROMANY(
			(PyObject*)Py_B, type, 1, 2,
			NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
	if (Py_X == NULL)
		return NULL;

	if (PyArray_DIM((PyArrayObject*)Py_X, 0) != N) {
		PyErr_SetString(PyExc_ValueError,
		                "b array has invalid shape");
		Py_DECREF(Py_X);
		return NULL;
	}

	if (csc) {
		if (NCFormat_from_spMatrix((SuperMatrix*)&A, N, N, nnz,
		                           (PyArrayObject *)nzvals, (PyArrayObject *)colind,
		                           (PyArrayObject *)rowptr, type)) {
			Py_DECREF(Py_X);
			return NULL;
		}
	}
	else {
		if (NRFormat_from_spMatrix((SuperMatrix*)&A, N, N, nnz, (PyArrayObject *)nzvals,
		                           (PyArrayObject *)colind, (PyArrayObject *)rowptr,
		                           type)) {
			Py_DECREF(Py_X);
			return NULL;
		}
	}

	if (DenseSuper_from_Numeric((SuperMatrix*)&B, (PyObject*)Py_X)) {
		Destroy_SuperMatrix_Store((SuperMatrix*)&A);
		Py_DECREF(Py_X);
		return NULL;
	}

	/* B and Py_X  share same data now but Py_X "owns" it */

	/* Setup options */

	jmpbuf_ptr = (volatile jmp_buf *)superlu_python_jmpbuf();
	SLU_BEGIN_THREADS;
	if (setjmp(*(jmp_buf*)jmpbuf_ptr)) {
		SLU_END_THREADS;
		goto fail;
	}
	else {
//		perm_c = intMalloc(N);
		perm_r = intMalloc(N);
		StatInit((SuperLUStat_t *) &stat);

		/* Compute direct inverse of sparse Matrix */
//		printf("calling zgssv_2\n");
		zgssv_2((superlu_options_t*)&options, (SuperMatrix*)&A, (int*)perm_c, (int*)perm_r,
		        (SuperMatrix*)&L, (SuperMatrix*)&U, (SuperMatrix*)&B, (SuperLUStat_t*)&stat,
		        (int*)&info);
//		printf("returned from zgssv_2\n");
		SLU_END_THREADS;
	}

	SUPERLU_FREE((void*)perm_r);
//	SUPERLU_FREE((void*)perm_c);
	Destroy_SuperMatrix_Store((SuperMatrix*)&A);	/* holds just a pointer to the data */
	Destroy_SuperMatrix_Store((SuperMatrix*)&B);
	Destroy_SuperNode_Matrix((SuperMatrix*)&L);
	Destroy_CompCol_Matrix((SuperMatrix*)&U);
	StatFree((SuperLUStat_t*)&stat);

//	printf("return\n");

	return Py_BuildValue("Ni", Py_X, info);

	fail:
	SUPERLU_FREE((void*)perm_r);
	SUPERLU_FREE((void*)perm_c);
	XDestroy_SuperMatrix_Store((SuperMatrix*)&A);	/* holds just a pointer to the data */
	XDestroy_SuperMatrix_Store((SuperMatrix*)&B);
	XDestroy_SuperNode_Matrix((SuperMatrix*)&L);
	XDestroy_CompCol_Matrix((SuperMatrix*)&U);
	XStatFree((SuperLUStat_t*)&stat);
	Py_XDECREF(Py_X);
	return NULL;
}

static PyObject *Py_gstrf(PyObject * self, PyObject * args,
			  PyObject * keywds)
{
    /* default value for SuperLU parameters */
    int N, nnz;
    PyArrayObject *rowind, *colptr, *nzvals;
    SuperMatrix A = { 0 };
    PyObject *result;
    PyObject *py_csc_construct_func = NULL;
    PyObject *option_dict = NULL;
    int type;
    int ilu = 0;

    static char *kwlist[] = { "N", "nnz", "nzvals", "colind", "rowptr",
        "csc_construct_func", "options", "ilu",
	NULL
    };

    int res =
	PyArg_ParseTupleAndKeywords(args, keywds, "iiO!O!O!O|Oi", kwlist,
				    &N, &nnz,
				    &PyArray_Type, &nzvals,
				    &PyArray_Type, &rowind,
				    &PyArray_Type, &colptr,
                                    &py_csc_construct_func,
				    &option_dict,
				    &ilu);

    if (!res)
	return NULL;

    if (!_CHECK_INTEGER(colptr) || !_CHECK_INTEGER(rowind)) {
	PyErr_SetString(PyExc_TypeError,
			"rowind and colptr must be of type cint");
	return NULL;
    }

    type = PyArray_TYPE((PyArrayObject*)nzvals);
    if (!CHECK_SLU_TYPE(type)) {
	PyErr_SetString(PyExc_TypeError,
			"nzvals is not of a type supported by SuperLU");
	return NULL;
    }

    if (NCFormat_from_spMatrix(&A, N, N, nnz, nzvals, rowind, colptr,
			       type)) {
	goto fail;
    }

    result = newSuperLUObject(&A, option_dict, type, ilu, py_csc_construct_func);
    if (result == NULL) {
	goto fail;
    }

    /* arrays of input matrix will not be freed */
    Destroy_SuperMatrix_Store(&A);
    return result;

  fail:
    /* arrays of input matrix will not be freed */
    XDestroy_SuperMatrix_Store(&A);
    return NULL;
}

static char gssv_doc[] =
    "Direct inversion of sparse matrix.\n\nX = gssv(A,B) solves A*X = B for X.";

static char gstrf_doc[] = "gstrf(A, ...)\n\
\n\
performs a factorization of the sparse matrix A=*(N,nnz,nzvals,rowind,colptr) and \n\
returns a factored_lu object.\n\
\n\
arguments\n\
---------\n\
\n\
Matrix to be factorized is represented as N,nnz,nzvals,rowind,colptr\n\
  as separate arguments.  This is compressed sparse column representation.\n\
\n\
N         number of rows and columns \n\
nnz       number of non-zero elements\n\
nzvals    non-zero values \n\
rowind    row-index for this column (same size as nzvals)\n\
colptr    index into rowind for first non-zero value in this column\n\
          size is (N+1).  Last value should be nnz. \n\
\n\
additional keyword arguments:\n\
-----------------------------\n\
options             specifies additional options for SuperLU\n\
                    (same keys and values as in superlu_options_t C structure,\n\
                    and additionally 'Relax' and 'PanelSize')\n\
\n\
ilu                 whether to perform an incomplete LU decomposition\n\
                    (default: false)\n\
";


/*
 * Main SuperLU module
 */

static PyMethodDef SuperLU_Methods[] = {
    {"gssv", (PyCFunction) Py_gssv, METH_VARARGS | METH_KEYWORDS,
     gssv_doc},
    {"gstrf", (PyCFunction) Py_gstrf, METH_VARARGS | METH_KEYWORDS,
     gstrf_doc},
    {"zgssv_1", (PyCFunction) Py_zgssv_1, METH_VARARGS | METH_KEYWORDS,
	 ""},
	 {"zgssv_2", (PyCFunction) Py_zgssv_2, METH_VARARGS | METH_KEYWORDS,
	  ""},
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_superlu",
    NULL,
    -1,
    SuperLU_Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__superlu(void)
{
    PyObject *m, *d;

    import_array();

    if (PyType_Ready(&SuperLUType) < 0) {
        return NULL;
    }

    if (PyType_Ready(&SuperLUGlobalType) < 0) {
    	return NULL;
    }

    m = PyModule_Create(&moduledef);
    d = PyModule_GetDict(m);

    Py_INCREF(&PyArrayFlags_Type);
    PyDict_SetItemString(d, "SuperLU",
			 (PyObject *) &SuperLUType);

    if (PyErr_Occurred())
	Py_FatalError("can't initialize module _superlu");

    return m;
}
