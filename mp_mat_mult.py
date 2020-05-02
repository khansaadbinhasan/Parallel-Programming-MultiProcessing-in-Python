"""
Program to multiply to mXn matrices with single core as well as multiple cores
The Program Generates two random matrices of given size, then multiplies them using single as well as multiple cores
the result from multiplciation via multiple cores is put into shared memory
"""

import os, sys, getopt
import multiprocessing 
import numpy as np
import time


def check_dims(matIn1, matIn2):
	"""
	function to check if dimensions of two matrices are compatible for multiplication
	input: two matrices matIn1(nXm) and matIn2(mXr)
	returns: Boolean whether dimensions compatible or not 
	"""

	m,n = matIn1.shape
	r,k = matIn2.shape

	if r == n:
		return True

	else:
		return False

def print_matrix_warning(matIn1, matIn2):
	"""
	function to print warning when dimensions of matrices are not compatible for multiplication
	input: two matrices matIn1(nXm) and matIn2(mXr)
	returns: None
	"""

	print("Matrix dims incorrect, please provide correct input matrices")
	print("Should be:")
	print("matIn1: nXm")
	print("matIn2: mXr")
	print("Dims provided: ")
	print("matIn1:", matIn1.shape[0],"X",matIn1.shape[1])
	print("matIn2:", matIn2.shape[0],"X",matIn2.shape[1])


def mat_mult(matIn1, matIn2): 
	""" 
	function to multiply two input matrices and returning the result
	input: two matrices matIn1(nXm) and matIn2(mXr)
	returns: matOut(nXr)
	"""


	if check_dims(matIn1, matIn2) == False:
		print_matrix_warning(matIn1, matIn2)
		return 	


	n,m = matIn1.shape
	_, r = matIn2.shape


	matOut = np.zeros((n,r),dtype=int)


	for i in range(n):
		for j in range(r):
			for k in range(m):
				matOut[i][j] = matOut[i][j] + matIn1[i][k]*matIn2[k][j]

	return matOut


def mat_mult_parallel(matIn1, matIn2, sharedMemArr, lastI):
	""" 
	function to multiply two input matrices and putting the result in shared memory 1D array
	input: two matrices matIn1(nXm) and matIn2(mXr), a 1D array of shared memory to put result in 
	returns: None
	"""

	if check_dims(matIn1, matIn2) == False:
		print_matrix_warning(matIn1, matIn2)
		return 	

	n,m = matIn1.shape
	_, r = matIn2.shape

	# print((n,m,r))

	for i in range(n):
		# print(i,j,k)
		for j in range(r):
			sumMat = 0
			for k in range(m):
				sumMat = sumMat + int(matIn1[i][k]*matIn2[k][j])

			# print(lastI)			
			sharedMemArr[lastI*r + i*r + j] = sumMat





def mat_trp(matIn):
	""" 
	function to transpose a matrix
	input: matrix matIn(nXm)
	returns: matrix matOut(mXn)
	"""

	n,m = matIn.shape
	matOut = np.zeros((m,n),dtype=int)

	for i in range(n):
		for j in range(m):
			matOut[j][i] = matIn[i][j]

	return matOut


def transfer_from_1D_to_2D_arr(arr1D, desired_shape):
	"""
	takes as input a 1D array and puts its content in a mXn 2D array
	input: arr1D !D array of mXn elements, desired_shape, a tuple (m,n) containing the desired shape of the 2D array
	returns: arr2D, 2D array containing mXn elements same as arr1D
	"""

	m,n = desired_shape
	ind = 0
	# print(m,n)
	arr2D = np.zeros((m,n),dtype=int)

	for i in range(m):
		for j in range(n):
			arr2D[i][j] = arr1D[ind]
			ind = ind + 1

	return arr2D


def runSequentialMatMul(matIn1, matIn2):
	"""
	takes as input two matrices and runs matrix multiplication, calculates time taken and prints the results and return the matrix multiplication
	input: Matrices matIn1(nXm), matIn2(nXm)
	"""

	timeStart = time.time()
	matOut = mat_mult(matIn1, mat_trp(matIn2))
	timeEnd = time.time()

	timeForExecSeq = (timeEnd - timeStart)*1000

	print("Time taken for sequential multiplication:",timeForExecSeq,"ms")

	return timeForExecSeq, matOut


def print_message_about_default(numProcessors, lowerLimit, upperLimit, n, m):
	"""
	prints message showing the argument values that will be used in the program
	input: None
	returns: None
	"""

	print("Values in arguments: ")
	print("numProcessors = %d"%numProcessors)
	print("lowerLimit = %d, upperLimit = %d"%(lowerLimit, upperLimit))
	print("n = %d, m = %d"%(n,m))

	print("If no input given then default values assumed.")


def print_instructions():
	"""
	prints instructions about usage of the program
	input: None
	returns: None
	"""

	print("Please give input like this or default values will be taken")
	print('python3 mp_mat_mult.py --numProcessors 4 --lowerLimit -10 --upperLimit 10 --n 100 --m 100')
	print("specify numProcessors you want to be used or all processors available will be used")
	print()

def get_args(argv):
	"""
	sets default values for parameters and takes input from the command line
	input: None
	returns: numProcessors, lowerLimit, upperLimit, n, m
	"""

	numProcessors = os.cpu_count()

	lowerLimit = -10
	upperLimit = 10

	n = 100
	m = 100


	print_instructions()

	try:
		opts, args = getopt.getopt(argv, "", ("numProcessors=", "lowerLimit=", "upperLimit=", "n=", "m="))
		
	except getopt.GetoptError:
		print_instructions()
		sys.exit(2)


	for opt, arg in opts:
		if opt == '--h' or opt == '-h':
			print_instructions()
			sys.exit()

		elif opt == "--numProcessors":
			if int(arg) <= numProcessors:
				numProcessors = int(arg)

		elif opt == "--lowerLimit":
			lowerLimit = int(arg)

		elif opt == "--upperLimit":
			upperLimit = int(arg)

		elif opt == "--m":
			m = int(arg)

		elif opt == "--n":
			n = int(arg)


	print_message_about_default(numProcessors, lowerLimit, upperLimit, n, m)


	return numProcessors, lowerLimit, upperLimit, n, m

def get_division(tot_rows, numProcessors):
	"""
	Takes in the number of rows in a matrix and the number of Processors available 
	for execution and gives a split of rows to be run on each processor
	input: tot_rows(int), the total number of rows in the matrix. numProcessors(int), the number of processors available for execution of task
	returns: A list containing the index of row to be used to slice the matrix
	"""


	division = []
	resLast, res = 0, 0
	division.append(res)

	while numProcessors != 0:
		if tot_rows >= numProcessors:
			resLast = resLast + res
			res = tot_rows // numProcessors
			
			division.append(resLast + res)
			tot_rows = tot_rows - res
			numProcessors = numProcessors - 1

		else:
			division = np.linspace(0,tot_rows,num=tot_rows, dtype=int)
			break

		
	return division


def runParallelMatMul(matIn1, matIn2, sharedMemArr, numProcessors):
	"""
	Executes the parallel multiplication of two matrices and puts the result in shared memory.
	input: Two matrices matIn1(nXm), matIn2(nXm) and an integer 1D array sharedMemArr(n*n elements) and number of processors available
	returns: timeForExecPar(float) the time to execute the parallel matrix multiplication, matOut(nXn) a matrix containing the final result
	"""

	procArr = []

	tot_rows, _ = matIn1.shape
	division = get_division(tot_rows, numProcessors)
	
	ind = multiprocessing.Value('i') 

	timeStart = time.time()

	for i in range(len(division)-1):
		matIn1_slice = matIn1[division[i] : division[i + 1], :]
		matIn2_trp = mat_trp(matIn2)

		p = multiprocessing.Process(target=mat_mult_parallel, args=(matIn1_slice, matIn2_trp, sharedMemArr, division[i])) 
		procArr.append(p)


	if len(division) == 1:
		sharedMemArr = mat_mult(matIn1, mat_trp(matIn2))


	for p in procArr:
		p.start() 

	for p in procArr:
		p.join() 

	timeEnd = time.time()

	timeForExecPar = (timeEnd - timeStart)*1000

	n = int(len(sharedMemArr)**0.5)
	desired_shape = (n,n)

	matOut = transfer_from_1D_to_2D_arr(sharedMemArr, desired_shape)	

	print("Time taken for parallel multiplication:",timeForExecPar,"ms")	

	return timeForExecPar, matOut

def main(argv):
	"""
	Main function to interface with the other functions available in the program
	input: argv, the inputs given via the command line
	returns: None
	"""
	
	numProcessors, lowerLimit, upperLimit, n, m = get_args(argv)

	matIn1 = np.random.randint(low = lowerLimit,high = upperLimit, size = (n,m))
	matIn2 = np.random.randint(low = lowerLimit,high = upperLimit, size = (n,m))

	print("\n"*3, "Input Matrices generated:")
	print("First Matrix",matIn1)
	print("\n"*2)
	print("Second Matrix",matIn2)
	print("\n"*2)

	timeForExecSeq, matOutSeq = runSequentialMatMul(matIn1, matIn2)
	sharedMemArr = multiprocessing.Array('i', n*n) 
	timeForExecPar, matOutPar = runParallelMatMul(matIn1, matIn2, sharedMemArr, numProcessors)


	print("\n"*2,"Output Matrix Generated For Serial Multiplication: ")
	print(matOutSeq)
	print("\n"*2,"Output Matrix Generated For Parallel Multiplication: ")
	print(matOutPar)
	print("\n"*2,"Hence, with {0} cores, parallel algorithm runs faster by {1}".format(numProcessors, float(timeForExecSeq/timeForExecPar)))
	print("\n"*2)

	## To check if multiplication is correct
	# start = time.time()
	# np_mult = np.dot(matIn1,mat_trp(matIn2))
	# end = time.time()
	# print((end-start)*1000)
	# print(np_mult == matOutPar)

if __name__ == "__main__": 
	argv = sys.argv[1:]
	main(argv)