/* Author: Ryan McAllister-Grum
 * Class: CSC484A Intro to Parallel Processing
 * Assignment: Midterm Option-B
 *
 * Usage: mpiexec -n <p> ./Geo-Mean <N>
 *    <p>: The number of parallel processes.
 *    <N>: The number of elements in the vector,
 *         between 1 and 8 inclusive.
 *
 * Description: The Geo-Mean program calculates the
 * the geometric mean of a given vector of integer
 * values by spreading the values out over the
 * processes, calculating their partial products,
 * recombining to calculate their total product,
 * and then taking the Nth root of the resulting
 * product, where 'N' is the number of values in
 * the vector, to find its geometric mean.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>


// Usage outputs an explanation of the program's command line parameters to stderr.
void Usage() {
	fprintf(stderr, "Usage: mpiexec -n <p> ./Geo-Mean <N>\n");
	fprintf(stderr, "<p>: The number of parallel processes.\n");
	fprintf(stderr, "<N>: The number of elements in the vector, between 1 and 8 inclusive.\n");
}

/* Get_Input reads in the user's input from the command line
 * and distributes it amongst all the processes.
 */
void Get_Input(int *n, int argc, char **argv, int my_rank, int comm_sz) {
	/* Prompt the user to enter in the
	 * vector length if they did not
	 * enter one in as a parameter.
	 */
	if (argc < 2) {
		if (my_rank == 0) {
			printf("Enter in the vector length: \n");
			scanf("%i", n);
		}
		// Distribute n to all processes.
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} else // Read in the vector length from the first parameter.
		*n = atoi(argv[1]);

	// Determine whether the length the user entered is valid.
	if (*n < 1 || *n > 8) {
		if (my_rank == 0) {
			fprintf(stderr, "Error, vector length %i is invalid.\n", *n);
			if (argc < 2) {
				fprintf(stderr, "Please enter an integer value between 1 and 8 inclusive.\n");
				fprintf(stderr, "Alternatively, you can start the program as follows:\n");
			}
			Usage();
		}
		MPI_Finalize();
		exit(0);
	}
}

/* Generate_And_Scatter_Vector creates and distributes a
 * vector of size n random int values evenly across the processes.
 */
void Generate_And_Scatter_Vector(int n, int comm_sz, int my_rank, int *vals, int **my_vals, int *my_n) {
	/* Create the local my_vals vector for each process to hold
	 * their chunk of values from the vals vector.
	 */
	*my_vals = (int *)malloc(sizeof(int) * (n/comm_sz +
		// Spreads the remainder out over the processes whose rank is smaller than the remainder.
		(my_rank <= (n - (n / comm_sz) * comm_sz) - 1 ? 1 : 0))
	);
	*my_n = n/comm_sz + 
		// Spreads the remainder out over the processes whose rank is smaller than the remainder.
		(my_rank <= (n - (n / comm_sz) * comm_sz) - 1 ? 1 : 0)
	;
	
	// Create vector of values that specifies how many elements each process receives.
	int *val_count = NULL;
	int *displs = NULL;
	
	// Initialize the vector of integer values.
	if (my_rank == 0) {
		// Create the temporary vector.
		vals = (int*) malloc(n*sizeof(int));
		// Seed the random number generator.
		srand(time(NULL));
		
		// Generate the random values.
		printf("Generated values:\n");
		int i;
		for (i = 0; i < n; i++) {
			// Geometric mean only accepts positive integer values.
			/* Per email correspondence with professor, generated
			 * integers are to be in the range 1 to 10.
			 */
			vals[i] = rand() % 10 + 1;
			if (i == n - 1)
				printf("%i\n", vals[i]);
			else
				printf("%i ", vals[i]);
		}
		
		
		// Space out output for easier reading.
		printf("\n");
		
		
		// Create vector of how many elements each process should receive
		// and the displacement value (starting index) for each chunk.
		val_count = (int *)malloc(sizeof(int) * comm_sz);
		displs = (int *)malloc(sizeof(int) * comm_sz);
		int displs_total = 0;
		for (i = 0; i < comm_sz; i++) {
			val_count[i] = n/comm_sz + (i <= (n - (n / comm_sz) * comm_sz) - 1 ? 1 : 0);
			displs[i] = (i == 0 ? 0 : displs_total);
			displs_total += n/comm_sz + (i <= (n - (n / comm_sz) * comm_sz) - 1 ? 1 : 0);
		}
	}

		
	// Distribute the vector into chunks amongst the processes.
	MPI_Scatterv(vals, val_count, displs, MPI_INT, *my_vals, *my_n, MPI_INT, 0, MPI_COMM_WORLD);
	
	// Clean up allocated vectors.
	free(vals);
	free(val_count);
	free(displs);
}

int main(int argc, char** argv) {
	// n holds the length of the vector.
	int n = 0;


	// Declare MPI variables and initialize MPI.
	int comm_sz, my_rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	
	
	// Fetch the user's input.
	Get_Input(&n, argc, argv, my_rank, comm_sz);
	
	
	// vals holds the original vector values.
	int *vals = NULL;
	/* my_vals holds the subset of vector
	 * values after being generated and
	 * distributed by process 0.
	 */
    int *my_vals;
	// my_n holds the length of the local vector.
	int my_n;
	
	
	// Create and scatter the vector of integers across the processes.
	Generate_And_Scatter_Vector(n, comm_sz, my_rank, vals, &my_vals, &my_n);
	
	
	// Start timing how long it takes to calculate the geometric mean.
	MPI_Barrier(MPI_COMM_WORLD);
	double local_start = MPI_Wtime();

	
	// Compute each process's partial product.
	int partial_product = 1;
	if (my_n > 0) {
		partial_product = my_vals[0];
		int i;
		for(i = 1; i < my_n; i++)
			partial_product *= my_vals[i];
	}
	
	
	// Reduce the partial products down to the final product.
	int total = 0;
	MPI_Reduce(&partial_product, &total, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);
	
	
	// Take the nth root to get the geometric mean.
	double geo_mean;
	if (my_rank == 0)
		geo_mean = pow((double) total, 1.0/n);


	// Stop the timer and calculate the elapsed time.
	double local_finish = MPI_Wtime();
	double local_elapsed = local_finish - local_start;
	double finish;
	MPI_Reduce(&local_elapsed, &finish, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	
	// Print the results.
	if (my_rank == 0) {
		printf("Total product of all elements is %i.\n", total);
		printf("Geometric mean is %g.\n", geo_mean);
		printf("\nElapsed time = %g seconds\n", finish);
	}
	
	
	// Clean up any remaining allocated pointers, MPI, and return.
	free(my_vals);	
	MPI_Finalize();
	return 0;
}