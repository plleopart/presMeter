#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int bool = 1;

double** create_2d_double_array(int arraySizeX, int arraySizeY) {
	double** theArray;
	theArray = (double**) malloc(arraySizeX * sizeof(double*));
	for (int i = 0; i < arraySizeX; i++)
		theArray[i] = (double*) malloc(arraySizeY * sizeof(double));
	return theArray;
}

void variance(void * in_1v, void * in_2v, void * in_3v, void * in_4v,
		void * in_5v, int n_row, int n_col, void * outputv) {

	double * in_1 = (double *) in_1v;
	double * in_2 = (double *) in_2v;
	double * in_3 = (double *) in_3v;
	double * in_4 = (double *) in_4v;
	double * in_5 = (double *) in_5v;
	double * output = (double *) outputv;

	double mean;
	double sum;
	int i;

	int x_cols = n_row;
	int y_rows = n_col;

	int x_counter;
	int y_counter;

	double * vector_matrix[5];

	vector_matrix[0] = in_1;
	vector_matrix[1] = in_2;
	vector_matrix[2] = in_3;
	vector_matrix[3] = in_4;
	vector_matrix[4] = in_5;

	for (x_counter = 0; x_counter < x_cols; x_counter++) {
		for (y_counter = 0; y_counter < y_rows; y_counter++) {

			mean = 0;
			sum = 0;
			for (i = 0; i < 5; i++) {
				mean += vector_matrix[i][(y_rows * x_counter) + y_counter];

			}
			mean /= 5;

			for (i = 0; i < 5; i++) {
				sum += pow(
						vector_matrix[i][(y_rows * x_counter) + y_counter]
								- mean, 2);

			}
			sum /= 5;
			output[(y_rows * x_counter) + y_counter] = sum;
		}
	}

}

void velocity_iteration(const void * partial_xv, const void * partial_yv,
		const void * partial_tv, const void * velocity_uv,
		const void * velocity_vv, int n_row, int n_col, double h_value,
		double alpha_value, double omega, void * outputv_x, void * outputv_y) {

	const double * partial_x = (double *) partial_xv;
	const double * partial_y = (double *) partial_yv;
	const double * partial_t = (double *) partial_tv;
	const double * velocity_u = (double *) velocity_uv;
	const double * velocity_v = (double *) velocity_vv;
	double * output_x = (double *) outputv_x;
	double * output_y = (double *) outputv_y;

	int row;
	int col;

	//here we iterate over the whole matrix
	for (row = 0; row < n_row; row++) {
		for (col = 0; col < n_col; col++) {
			double J_11 = partial_x[(n_col * row) + col]
					* partial_x[(n_col * row) + col];
			double J_12 = partial_x[(n_col * row) + col]
					* partial_y[(n_col * row) + col];
			double J_13 = partial_x[(n_col * row) + col]
					* partial_t[(n_col * row) + col];

			double J_21 = partial_y[(n_col * row) + col]
					* partial_x[(n_col * row) + col];
			double J_22 = partial_y[(n_col * row) + col]
					* partial_y[(n_col * row) + col];
			double J_23 = partial_y[(n_col * row) + col]
					* partial_t[(n_col * row) + col];

			double second_term_numerator_u = ((h_value * h_value) / alpha_value)
					* (J_12 * velocity_v[(n_col * row) + col] + J_13);
			double denominator_u = ((h_value * h_value) / alpha_value) * J_11;

			double second_term_numerator_v = ((h_value * h_value) / alpha_value)
					* (J_21 * velocity_u[(n_col * row) + col] + J_23);
			double denominator_v = ((h_value * h_value) / alpha_value) * J_22;

			double neigh_u = 0;
			double neigh_v = 0;
			int count;
			if (row > 0 && row < n_row && col > 0 && col < n_col) {
				//calculate neighbors of u and v

				for (count = 0; count < 8; count++) {
					switch (count) {
					case 0:
						neigh_u += velocity_u[(n_col * (row - 1)) + col - 1];
						neigh_v += velocity_v[(n_col * (row - 1)) + col - 1];
						break;
					case 1:
						neigh_u += velocity_u[(n_col * (row - 1)) + col];
						neigh_v += velocity_v[(n_col * (row - 1)) + col];
						break;
					case 2:
						neigh_u += velocity_u[(n_col * (row - 1)) + col + 1];
						neigh_v += velocity_v[(n_col * (row - 1)) + col + 1];
						break;
					case 3:
						neigh_u += velocity_u[(n_col * row) + col - 1];
						neigh_v += velocity_v[(n_col * row) + col - 1];
						break;
					case 4:
						neigh_u += velocity_u[(n_col * (row + 1)) + col + 1];
						neigh_v += velocity_v[(n_col * (row + 1)) + col + 1];
						break;
					case 5:
						neigh_u += velocity_u[(n_col * (row + 1)) + col];
						neigh_v += velocity_v[(n_col * (row + 1)) + col];
						break;
					case 6:
						neigh_u += velocity_u[(n_col * (row + 1)) + col - 1];
						neigh_v += velocity_v[(n_col * (row + 1)) + col - 1];
						break;
					case 7:
						neigh_u += velocity_u[(n_col * row) + col - 1];
						neigh_v += velocity_v[(n_col * row) + col - 1];
						break;

					}

				}

				//calculate the final value of the matrix
				output_x[(n_col * row) + col] = ((1 - omega)
						* velocity_u[(n_col * row) + col])
						+ (omega
								* ((neigh_u - second_term_numerator_u)
										/ (8 + denominator_u)));

				output_y[(n_col * row) + col] = ((1 - omega)
						* velocity_v[(n_col * row) + col])
						+ (omega
								* ((neigh_v - second_term_numerator_v)
										/ (8 + denominator_v)));

			}
		}

	}

//printf("C: Multiplication done...\n");
}

void reduce_matrix(const void * indatav, int x_col, int y_row, int x_sub,
		int y_sub, void * outdatav) {

	const double * indata = (double *) indatav;
	double * outdata = (double *) outdatav;

	int x_size = x_col;
	int y_size = y_row;

	int reduced_x = x_sub;
	int reduced_y = y_sub;

	int size_sub_x = x_size / reduced_x;
	int size_sub_y = y_size / reduced_y;

	//double** input_matrix = create_2d_double_array(x_size, y_size);
	//double** reduced_matrix = create_2d_double_array(reduced_x, reduced_y);
	int x;
	int y;
	/*
	for (x = 0; x < x_size; x++) {
		for (y = 0; y < y_size; y++) {
			input_matrix[x][y] = indata[(y_size * x) + y];
		}
	}
	*/
	for (x = 0; x < x_size; x++) {
		for (y = 0; y < y_size; y++) {
			//reduced_matrix[(x / size_sub_x)][(y / size_sub_y)] += input_matrix[x][y];
			outdata[(reduced_y * (x / size_sub_x)) + (y / size_sub_y)] += (indata[(y_size * x) + y])/((x_size / reduced_x) * (y_size / reduced_y));
		}
	}
	/*
	for (x = 0; x < reduced_x; x++) {
		for (y = 0; y < reduced_y; y++) {
			//reduced_matrix[x][y] /= (x_size / reduced_x) * (y_size / reduced_y);
			outdata[(reduced_y * x) + y] /= (x_size / reduced_x) * (y_size / reduced_y);
			//outdata[(reduced_y * x) + y] = reduced_matrix[x][y];
			//printf("[%.2f]", reduced_matrix[x][y]);
		}
		//printf("\n");
	}
	*/
	/*
	int i;
	for (i = 0; i < x_size; i++) {
		free(input_matrix[i]);
	}
	free(input_matrix);
	for (i = 0; i < reduced_x; i++) {
		free(reduced_matrix[i]);
	}
	free(reduced_matrix);
	*/
}

//input of this matrix is the main matrix, the kernel matrix, the number of rows
//from main matrix, the number of cols from main matrix, the number of rows from
//kernel matrix, the number of cols from kernel matrix and the matrix where we want
//to put the result.
//This resulting matrix must be initialized before from python
void convol_2d(const void * indatav, const void * indatav_kernel, int rowcount,
		int colcount, int rowcount_kernel, int colcount_kernel, void * outdatav) {

	const double * indata = (double *) indatav;
	const double * indata_kernel = (double *) indatav_kernel;
	double * outdata = (double *) outdatav;

//counters for input x and y iterations
	int in_x;
	int in_y;

//counter for kernel x and y iterations
	int ker_x;
	int ker_y;

//flipped index of rows and cols
	int flipped_ker_x;
	int flipped_ker_y;

//index used to check boundary conditions
	int b_in_x;
	int b_in_y;

//size of input matrix
	int in_x_size = rowcount;
	int in_y_size = colcount;

//size of kernel matrix
	int ker_x_size = rowcount_kernel;
	int ker_y_size = colcount_kernel;

//calculation of the mid point of the kernel matrix for x and y
//we use this to determine where to start computing
	int center_ker_x = round(ker_x_size / 2);
	int center_ker_y = round(ker_y_size / 2);

//input matrix and kernel matrix definitions
	double** input_matrix = create_2d_double_array(in_x_size, in_y_size);
	double** kernel_matrix = create_2d_double_array(ker_x_size, ker_y_size);

	//here we transform the one dimensional array into a two dimensional one
	for (in_x = 0; in_x < in_x_size; in_x++) {
		for (in_y = 0; in_y < in_y_size; in_y++) {
			input_matrix[in_x][in_y] = indata[(in_y_size * in_x) + in_y];
		}
	}

	for (ker_x = 0; ker_x < ker_x_size; ker_x++) {
		for (ker_y = 0; ker_y < ker_y_size; ker_y++) {
			kernel_matrix[ker_x][ker_y] = indata_kernel[(ker_y_size * ker_x)
					+ ker_y];
		}
	}
	
	//this loop iterate over input x values, so rows
	for (in_x = 0; in_x < in_x_size; in_x++) {
		//this loop iterate over input y values, so cols
		for (in_y = 0; in_y < in_y_size; in_y++) {
			//this loop iterate over kernel x values, so rows
			for (ker_x = 0; ker_x < ker_x_size; ker_x++) {
				//here we flip index of rows and we put the value to flipped_ker_x
				flipped_ker_x = ker_x_size - 1 - ker_x;

				//this loop iterate over kernel y values, so cols
				for (ker_y = 0; ker_y < ker_y_size; ker_y++) {
					//here we flip index of cols and we put the value to flipped_ker_y
					flipped_ker_y = ker_y_size - 1 - ker_y;

					//calculate index to check boundary conditions
					b_in_x = in_x + (ker_x - center_ker_x);
					b_in_y = in_y + (ker_y - center_ker_y);

					//check boundary conditions
					if (b_in_x >= 0 && b_in_x < in_x_size && b_in_y >= 0
							&& b_in_y < in_y_size) {
	
						outdata[(in_y_size * in_x) + in_y] =
								outdata[(in_y_size * in_x) + in_y]
										+ (input_matrix[b_in_x][b_in_y]
												* kernel_matrix[flipped_ker_x][flipped_ker_y]);

					}

				}
			}

		}

	}

	//free memory
	int i;
	for (i = 0; i < in_x_size; i++) {
		free(input_matrix[i]);
		//free(output_matrix[i]);
	}
	free(input_matrix);
	//free(output_matrix);

	for (i = 0; i < ker_x_size; i++) {
		free(kernel_matrix[i]);
	}
	free(kernel_matrix);

	//printf("C: Convolution done...\n");

}
