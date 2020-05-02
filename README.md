## Matrix Multiplication
Program to multiply to mXn matrices with single core as well as multiple cores. The Program Generates two random matrices of given size, then multiplies them using single as well as multiple cores the result from multiplciation via multiple cores is put into shared memory.

### How to Use
To use this program, `git clone` this repo and then `cd` to the folder containing this repo. then run:
```
python mp_mat_mult.py --numProcessors 4 --lowerLimit -10 --upperLimit 10 --n 100 --m 100
```
or simply, to run with default arguments
```
python mp_mat_mult.py
```


`numProcessors`: Number of Processors you want to be used, By default all the processors available will be used. If more than available processors are given, default processors will be used.

`lowerLimit`: The lower limit integer to which the generated input matrice will adhere to, default is set to -10.

`upperLimit`: The upper limit integer to which the generated input matrice will adhere to, default is set to 10.

`n`: Number of rows in input matrices.

`m`: Number of columns in input matrices.


### Results
The Program outputs the results of matrix multiplication with both sequential as well as parallel multiplication. It also outputs the time taken(in ms) to execute the two multiplications and how fast is the implementation of parallel multiplication with respect to sequential multiplication.


## Parallel Scan
Program to perform inclusive and exclusive scans. Implements sequential scans for inclusive and exclusive scans as well as parallel scans for inclusive(hillis-steeles) and exclusive(blelloch). Blelloch scan is not correctly and fully implemented since, its implementation is based upon a single example: [1,2,3,4,5,6,7,8], hence does not work on other examples, though may work on examples containing multiple of 8 elements.

## How to use
To use this program, `git clone` this repo and then `cd` to the folder containing this repo. then run:
```
python parallel_scan.py
```
This should open a prompt asking to input a list or to fill in a random list e.g,
```
Input List/Random List(i/any other key): i
Enter List: 1 2 3 4 5 6 7 8
[1, 2, 3, 4, 5, 6, 7, 8]
```
or
```
Input List/Random List(i/any other key): 
Enter size of list desired: 23
[4, 5, 8, 1, -10, -1, 1, 3, 5, -2, 7, 1, 3, 2, 1, 8, -7, 7, 8, -9, -9, 4, -1]
```

## Results
This will print the results of different scans