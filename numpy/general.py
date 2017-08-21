import numpy as np

# an example
a = np.arange(15).reshape(3,5)
print(a)
print(a.shape)
print(a.ndim)
print(a.dtype.name)
print(a.itemsize)
print(a.size)
print(type(a))

b = np.array([6,7,8])
print(type(b))

# Array Creation
# There are several ways to create arrays.
# a = np.array(1,2,3,4)    # WRONG
# a = np.array([1,2,3,4])  # RIGHT
# array transforms sequences of sequences into two-dimensional arrays, sequences of sequences of sequences into three-dimensional arrays, and so on.

b = np.array([(1.5,2,3), (4,5,6)])
print(b)

#The type of the array can also be explicitly specified at creation time:
c = np.array( [ [1,2], [3,4] ], dtype=complex )
print(c)

# The function zeros creates an array full of zeros, the function ones creates an array full of ones
print(np.zeros((3,4)))

print(np.ones( (2,3,4), dtype=np.int16 ))

print(np.empty( (2,3) ))                               # uninitialized, output may vary

# To create sequences of numbers, NumPy provides a function analogous to range that returns arrays instead of lists.
print(np.arange(10,30,5))

print(np.linspace( 0, 2, 9 ) )

# Basic Operations
# Arithmetic operators on arrays apply elementwise. A new array is created and filled with the result.

a = np.array([20,30,40,50])
b = np.arange(4)
c = a - b
print(c)

#Unlike in many matrix languages, the product operator * operates elementwise in NumPy arrays. The matrix product can be performed using the dot function or method:
A = np.array([[1,1], [0,1]])
B = np.array([[2,0],[3,4]])
print(A * B)
print(A.dot(B))
print(np.dot(A, B))

# Some operations, such as += and *=, act in place to modify an existing array rather than create a new one.
# Many unary operations, such as computing the sum of all the elements in the array, are implemented as methods of the ndarray class.
a = np.random.random((2,3))
print(a)
print(a.sum())
print(a.min())
print(a.max())

# By default, these operations apply to the array as though it were a list of numbers, regardless of its shape. However, by specifying the axis parameter you can apply an operation along the specified axis of an array:
b = np.arange(12).reshape(3,4)
print(b)
print(b.sum(axis=0))
print(b.min(axis=1))
print(b.cumsum(axis=1))

# Universal Functions
# NumPy provides familiar mathematical functions such as sin, cos, and exp.
B = np.arange(3)
print(B)
print(np.exp(B))
print(np.sqrt(B))

# Indexing, Slicing and Iterating
# skipped see https://docs.scipy.org/doc/numpy-dev/user/quickstart.html


# Shape Manipulation
a = np.floor(10*np.random.random((3,4)))
print(a)
print(a.ravel())  # returns the array, flattened
print(a.reshape(6,2))
# print(a.reshape(6,3)) # Value Error

# Stacking together different arrays
# skipped
# Splitting one array into several smaller ones
# View or Shallow Copy
# Deep Copy
# The copy method makes a complete copy of the array and its data.

