# Reference Guide for Machine Learning Code Implementation

This document provides detailed mathematical formulas and equations necessary for implementing machine learning algorithms.

---

## 1. Linear Regression

### Hypothesis Function:
```math
h(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
```

Where:
```math
\mathbf{w} \text{ is the weight vector}
```
```math
\mathbf{x} \text{ is the feature vector}
```
```math
b \text{ is the bias (intercept)}
```

### Cost Function:
```math
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^m \left( h(\mathbf{x}^{(i)}) - y^{(i)} \right)^2
```

Where:
```math
m \text{ is the number of training examples}
```
```math
h(\mathbf{x}^{(i)}) \text{ is the predicted output for the } i \text{-th example}
```
```math
y^{(i)} \text{ is the actual output for the } i \text{-th example}
```

### Gradient Descent:
```math
\frac{\partial J}{\partial \mathbf{w}_j} = \frac{1}{m} \sum_{i=1}^m \left( h(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}
```
```math
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m \left( h(\mathbf{x}^{(i)}) - y^{(i)} \right)
```

Update rules:
```math
\mathbf{w}_j \leftarrow \mathbf{w}_j - \alpha \frac{\partial J}{\partial \mathbf{w}_j}
```
```math
b \leftarrow b - \alpha \frac{\partial J}{\partial b}
```

Where:
```math
\alpha \text{ is the learning rate}
```

---

# FastAPI Reference Guide

## FastAPI

### Overview
FastAPI is a modern, high-performance web framework for building APIs with Python 3.6+ based on standard Python type hints. It is designed to be fast and easy to use, leveraging Python's asyncio for asynchronous operations.

### Basic Application Structure
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

In this example, `item_id` is extracted from the URL path, and an optional query parameter `q` can also be included in the request.

### Running the Application
```bash
uvicorn main:app --reload --port 8080 --host 0.0.0.0
```

### Key Features
- **Type Hints:** Utilizes Python type hints for data validation and serialization.
- **Automatic Documentation:** Generates interactive API documentation with Swagger UI and ReDoc.
- **Asynchronous Support:** Built-in support for asynchronous request handling.

### Defining a POST Endpoint with Data Validation
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

### Documentation Access
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

This guide provides foundational structures and concepts for FastAPI.

# NumPy and Pandas Reference Guide

## NumPy

### Overview
NumPy is a powerful library for numerical computing in Python. It provides support for multi-dimensional arrays, mathematical functions, and linear algebra operations.

### Importing NumPy
```python
import numpy as np
```

### Key Features

#### Creating Arrays
```python
# Create a 1D array
array_1d = np.array([1, 2, 3])

# Create a 2D array
array_2d = np.array([[1, 2], [3, 4]])

# Create an array of zeros
zeros = np.zeros((3, 3))

# Create an array of ones
ones = np.ones((2, 2))

# Create an array with a range of values
range_array = np.arange(0, 10, 2)

# Create a linearly spaced array
linspace_array = np.linspace(0, 1, 5)
```

#### Array Operations
```python
# Element-wise addition
result = array_1d + 2

# Element-wise multiplication
result = array_1d * 2

# Dot product
dot_product = np.dot(array_1d, [4, 5, 6])

# Transpose
transposed = array_2d.T
```

#### Indexing and Slicing
```python
# Accessing elements
first_element = array_1d[0]

# Slicing
subset = array_2d[0:2, 1:2]
```

#### Mathematical Functions
```python
# Mean and standard deviation
mean_value = np.mean(array_1d)
std_dev = np.std(array_1d)

# Trigonometric functions
sine_values = np.sin(array_1d)

# Summation
sum_value = np.sum(array_2d)
```

---

## Pandas

### Overview
Pandas is a versatile library for data manipulation and analysis. It provides easy-to-use data structures like DataFrames and Series.

### Importing Pandas
```python
import pandas as pd
```

### Key Features

#### Creating Data Structures
```python
# Create a Series
series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
}
dataframe = pd.DataFrame(data)
```

#### Reading and Writing Data
```python
# Read a CSV file
df = pd.read_csv('file.csv')

# Write to a CSV file
df.to_csv('output.csv', index=False)
```

#### Basic Operations
```python
# Access columns
ame_column = dataframe['Name']

# Access rows by indexirst_row = dataframe.iloc[0]

# Access rows by condition
filtered_rows = dataframe[dataframe['Age'] > 30]
```

#### Data Cleaning
```python
# Drop missing values
cleaned_df = dataframe.dropna()

# Fill missing values
filled_df = dataframe.fillna(0)
```

#### Data Aggregation
```python
# Grouping data
grouped = dataframe.groupby('Age').mean()

# Summarizing data
summary = dataframe.describe()
```

---

This guide provides a quick reference for using NumPy and Pandas for data analysis and numerical computing. For more advanced features, refer to the official documentation:
- NumPy: [https://numpy.org/doc/](https://numpy.org/doc/)
- Pandas: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
