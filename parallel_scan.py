"""
Program to perform inclusive and exclusive scans. 
Implements sequential scans for inclusive and exclusive scans
as well as parallel scans for inclusive(hillis-steeles) and exclusive(blelloch)
"""

import numpy as np
import multiprocessing
import os

def sequential_scan_inclusive(array):
	"""
	Takes in an array, performs inclusive scan sequentially, returns the array with the results
	input: 1D array	of n elements.
	returns: 1D array, summed_arr of n elements.
	"""

	(n,) = array.shape

	summed_arr = np.zeros((n,),dtype=int)


	summed_arr[0] = array[0]

	for i in range(n-1):
		summed_arr[i+1] = array[i+1] + summed_arr[i]

	return summed_arr


def sequential_scan_exclusive(array):
	"""
	Takes in an array, performs exclusive scan sequentially, returns the array with the results
	input: 1D array	of n elements.
	returns: 1D array, summed_arr of n elements.
	"""

	(n,) = array.shape

	summed_arr = np.zeros((n,),dtype=int)

	for i in range(n-1):
		summed_arr[i+1] = array[i] + summed_arr[i]


	return summed_arr

def hillis_steeles_loop(j, prev_list, newArr, to_update):
	"""
	Main loop of hillis steeles algo for updation
	input: j(int) the step of the algo, prev_list(n elem list) containing elements from last iteration, 
			newArr(array n elem) shared memory to which we have to write , to_update(tuple) containing indices (ind1, ind2)
	returns: None
	"""

	n = len(prev_list)
	ind1, ind2 = to_update

	if ind1 < 2**j:
		ind1 = 2**j

	for i in range(ind1, ind2):
		newArr[i] = prev_list[i] + prev_list[ i - 2**j ]


def pad_if_needed(inp_list):
	"""
	Checks the number of elements in the input list, if number is odd then one element is added
	input: inp_list, list containing n elements
	returs: inp_list, list containing n or n+1 elements
	"""

	n = len(inp_list)

	if n % 2 == 0:
		return inp_list, False

	else:
		inp_list.append(0)
		return inp_list, True

def get_division(numElements, numProcessors):
	"""
	Takes in the number of elements in a List and the number of Processors available 
	for execution and gives a split of elements to be run on each processor
	input: numElements(int), the total number of elements in the list. numProcessors(int), the number of processors available for execution of task
	returns: A list containing the index of list to be used to slice the list
	"""

	division = []
	resLast, res = 0, 0
	division.append(res)

	while numProcessors != 0:
		if numElements >= numProcessors:
			resLast = resLast + res
			res = numElements // numProcessors
			
			division.append(resLast + res)
			numElements = numElements - res
			numProcessors = numProcessors - 1

		else:
			division = np.linspace(0,numElements+1,num=numElements+1, dtype=int)
			break

		
	return division


def hillis_steeles_scan_inclusive_parallel(inp_list):
	"""
	Takes in a list, performs hillis-steeles inclusive scan parallelly, returns a shared memory array with the results
	input: inp_list, 1D list of n elements.
	returns: 1D Array, newArr of n elements.
	"""
	numProcessors = os.cpu_count()

	inp_list, removeLast = pad_if_needed(inp_list)

	n = len(inp_list)

	prev_list = list(inp_list)
	newArr = multiprocessing.Array('i',n)
	newArr[:] = inp_list	


	for j in range(int(np.ceil(np.log2(n)))):

		procArr = []

		division = get_division(n - 2**j, numProcessors)

		for i in range(len(division)-1):

			ind1, ind2 = division[i] + 2**j - 1, division[i + 1] + 2**j 

			if ind2 > len(prev_list):
				ind2 = len(prev_list) 

			p = multiprocessing.Process(target=hillis_steeles_loop, args=(j, prev_list, newArr, (ind1, ind2))) 
			procArr.append(p)

			
		for p in procArr:
			p.start() 

		for p in procArr:
			p.join() 


		prev_list = list(newArr)	

	if removeLast:
		newArr = newArr[:-1]

	return newArr




# def refill_list(some_list):
# 	if( len(some_list) > 1 and (some_list[0] - some_list[1] == 1) ):
# 		return some_list

# 	new_list = []

# 	while len(some_list) > 0:

# 		elem1 = some_list.pop()
		

# 		if len(some_list) == 0:
# 			elem2 = 0

# 		else:
# 			elem2 = some_list[-1]
			
# 		intMean = ( elem1 + elem2 ) // 2

# 		new_list.insert(0,elem1)
# 		new_list.insert(0,intMean)

# 		# print(new_list)

	
# 	return list(set(new_list))


# def blelloch_scan_exclusive_parallel(inp_list):
# 	"""
# 	Takes in an list, performs blelloch exclusive scan parallely, returns the list with the results
# 	input: 1D list	of n elements.
# 	returns: 1D list, summed_arr of n elements.
# 	"""

# 	inp_list, removeLast = pad_if_needed(inp_list)

# 	n = len(inp_list)

# 	summed_list = list(inp_list)
# 	new_list = list(inp_list)

# 	ind1_bar = 0
# 	ind2_bar = 1

# 	# Reduce Steps
# 	for j in range(int(np.ceil(np.log2(n)))):
# 		ind1 = ind1_bar
# 		ind2 = ind2_bar
		
# 		for i in range(n // 2**(j+1)):

# 			summed_list[ind2] = new_list[ind2] + new_list[ind1]

# 			ind1 = ind1 + 2**(j+1)
# 			ind2 = ind1 + 2**j

# 		new_list = summed_list			

# 		ind1_bar = ind1_bar + 2**j
# 		ind2_bar = ind1_bar + 2**(j+1)


# 	ind1_bar = ( n - 1 ) // 2
# 	ind2_bar = n - 1 


# 	elemList = [ind1_bar, ind2_bar]

# 	summed_list[-1] = 0

# 	# print(summed_list)

# 	# Downsweep
# 	for j in range(int(np.ceil(np.log2(n)))):
# 		new_list = list(elemList)

# 		while new_list:
# 			ind2 = new_list.pop()

# 			if not new_list:
# 				ind1 = 0

# 			else:
# 				ind1 = new_list.pop()

# 			temp = summed_list[ind1]
# 			summed_list[ind1] = summed_list[ind2]
# 			summed_list[ind2] = temp + summed_list[ind2]

# 		print(summed_list)
# 		elemList = refill_list(elemList)


# 	if removeLast:
# 		summed_list.pop()

# 	return summed_list


def main():
	inpResp = input("Input List/Random List(i/any other key): ")

	if inpResp == 'i':
		print("Enter List: ", end="")
		inpList = [int(x) for x in input().split()]

	else:
		lenth = 10
		lenth = int(input("Enter size of list desired: "))
		inpList = list(np.random.randint(low = -10,high = 10, size = (lenth)))


	# inpList = [1,2,3,4,5,6,7,8,9]

	summed_arr_inclusive_sequential = sequential_scan_inclusive(np.array(inpList))
	summed_arr_exclusive_sequential = sequential_scan_exclusive(np.array(inpList))
	summed_arr_hillis_steeles_inclusive_parallel = hillis_steeles_scan_inclusive_parallel(list(inpList))

	# summed_arr_blelloch_exclusive_parallel, time_par_exc = blelloch_scan_exclusive_parallel(inpList)
	
	print("input List:", inpList)
	print("sequential inclusive:", summed_arr_inclusive_sequential)
	print("sequential exclusive:", summed_arr_exclusive_sequential)	
	print("parallel inclusive hillis-steele:", summed_arr_hillis_steeles_inclusive_parallel[:])
	# print("parallel exclusive blelloch:", summed_arr_blelloch_exclusive_parallel)	



if __name__ == '__main__':
	main()