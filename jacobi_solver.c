/* Code for the Jacobi method of solving a system of linear equations
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 22, 2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm
*/

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "jacobi_solver.h"
#include <pthread.h>

/* Uncomment the line below to spit out debug information */
/* #define DEBUG */

pthread_barrier_t my_barrier;
int done;

/* Data structure defining arguments to pass to each worker thread */
typedef struct thread_data_s {
    int tid;                        /* The thread ID */
    int num_threads;                /* Number of threads in the pool */
    int num_elements;               /* Number of elements in the vector */
    matrix_t A;                     /* Matrix A */
    matrix_t B;                     /* Matrix B */
    matrix_t x;                     /* Matrix x */
    matrix_t new_x;                 /* Matrix new_x */
} thread_data_t;

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix-size [num-threads]\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
        fprintf(stderr, "num-threads: number of threads to utilize\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    int num_threads;
    if (argc == 3)
        num_threads = atoi(argv[2]);
    else
        num_threads = 4;

    matrix_t  A;                    /* N x N constant matrix */
    matrix_t  B;                    /* N x 1 b matrix */
    matrix_t reference_x;           /* Reference solution */
    matrix_t mt_solution_x;         /* Solution computed by pthread code */

	struct timeval start, stop;

    /* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
    srand(time(NULL));
    A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
    if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
    }

    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
    reference_x = allocate_matrix(matrix_size, 1, 0);
    mt_solution_x = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
    print_matrix(A);
    print_matrix(B);
    print_matrix(reference_x);
#endif

    /* Compute Jacobi solution using reference code */
    fprintf(stderr, "\nGenerating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
	gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Compute the Jacobi solution using pthreads.
     * Solutions are returned in mt_solution_x.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads\n");
	gettimeofday(&start, NULL);
    compute_using_pthreads(A, mt_solution_x, B, num_threads);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    free(A.elements);
    free(B.elements);
    free(reference_x.elements);
    free(mt_solution_x.elements);

    exit(EXIT_SUCCESS);
}

/* Jacobi worker function */
void *jacobi(void *args){
    thread_data_t *thread_data = (thread_data_t *)args; /* Typecast argument to pointer to thread_data_t structure */

    int i, j;
    int num_iter = 0;
    double ssd, mse;

    while (!done) {
        for (i = thread_data->tid; i < thread_data->num_elements; i+=thread_data->num_threads) {
            double sum =  sum = -thread_data->A.elements[i * thread_data->num_elements + i] * thread_data->x.elements[i];
            for (j = 0; j < thread_data->num_elements; j++) {
                sum += thread_data->A.elements[i * thread_data->num_elements + j] * thread_data->x.elements[j];
            }

            /* Update values for the unkowns for the current row. */
            thread_data->new_x.elements[i] = (thread_data->B.elements[i] - sum)/thread_data->A.elements[i * thread_data->num_elements + i];
        }
        if(pthread_barrier_wait(&my_barrier));
        if(thread_data->tid == 0){
            ssd = 0.0;
            for (i = 0; i < thread_data->num_elements; i++) {
                ssd += (thread_data->new_x.elements[i] - thread_data->x.elements[i]) * (thread_data->new_x.elements[i] - thread_data->x.elements[i]);
                thread_data->x.elements[i] = thread_data->new_x.elements[i];
            }
            num_iter++;
            mse = sqrt(ssd); /* Mean squared error. */
            #ifdef DEBUG
                fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse);
            #endif

            if (mse <= THRESHOLD){
                done = 1;
            fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
            }
        }
        if(pthread_barrier_wait(&my_barrier));
    }

}

/* Result must be placed in mt_sol_x. */
void compute_using_pthreads (const matrix_t A, matrix_t mt_sol_x, const matrix_t B, int num_threads)
{
    int i;
    int num_elements = A.num_rows;
    pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));    /* Data structure to store thread IDs */
    pthread_attr_t attributes;                                                      /* Thread attributes */
    pthread_attr_init (&attributes);                                                /* Initialize thread attributes to default values */

    /* Allocate n x 1 matrix to hold iteration values.*/
    matrix_t new_x = allocate_matrix(num_elements, 1, 0);

    /* Initialize current jacobi solution. */
    for (i = 0; i < num_elements; i++)
        mt_sol_x.elements[i] = B.elements[i];

    /* Perform Jacobi iteration. */
    done = 0;

    pthread_barrier_init(&my_barrier, NULL, num_threads);

    /* Fork point: Allocate memory for required data structures and create the worker threads */
    thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid          = i;
        thread_data[i].num_threads  = num_threads;
        thread_data[i].num_elements = num_elements;
        thread_data[i].A            = A;
        thread_data[i].B            = B;
        thread_data[i].x 	        = mt_sol_x;
        thread_data[i].new_x 	    = new_x;

        pthread_create(&thread_id[i], &attributes, jacobi, (void *)&thread_data[i]);
    }

    /* Join point: Wait for the workers to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(thread_id[i], NULL);

    /* Free data structures */
    free((void *)thread_data);
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;

    M.elements = (float *)malloc(size * sizeof(float));
    for (i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }

    return M;
}

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
    for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
            fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
        }

        fprintf(stderr, "\n");
    }

    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
    return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
    float diag_element;
    float sum;
    for (i = 0; i < M.num_rows; i++) {
        sum = 0.0;
        diag_element = M.elements[i * M.num_rows + i];
        for (j = 0; j < M.num_columns; j++) {
            if (i != j)
                sum += abs(M.elements[i * M.num_rows + j]);
        }

        if (diag_element <= sum)
            return -1;
    }

    return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix (int num_rows, int num_columns)
{
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
    fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
    for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);

    /* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
    for (i = 0; i < num_rows; i++) {
        row_sum = 0.0;
        for (j = 0; j < num_columns; j++) {
            row_sum += fabs(M.elements[i * M.num_rows + j]);
        }

        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
    }

    /* Check if matrix is diagonal dominant */
    if (check_if_diagonal_dominant(M) < 0) {
        free(M.elements);
        M.elements = NULL;
    }

    return M;
}



