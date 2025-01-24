# Introduction to Python

## Course Overview

This course introduces the fundamentals of Python programming, covering core concepts such as:

- Math Expressions
- Strings
- Variables
- Logic/Boolean
- Loops
- Functions
- Lists
- Dictionaries
- Modules

## What is Programming?

Programming involves writing instructions to solve problems and make computers useful. It powers various applications such as:

- Operating Systems
- Email Clients
- Web Browsers
- Games
- Satellites

### Natural vs Programming Languages
- Natural languages (e.g., English) are used for human communication.
- Programming languages (e.g., Python) are used to instruct computers.

## Why Python?

Python is a popular choice for programming due to its:

- **Beginner-friendliness**: Easy to learn and use.
- **Readable syntax**: Feels like writing logic in English.
- **Extensive community**: Huge open-source community with plenty of resources.
- **Versatile libraries** for:
  - Data Science
  - Machine Learning
  - Data Analysis
  - Web Development
  - Automation
  - Gaming and more...

## Python Development Environment

An Integrated Development Environment (IDE) is software that provides comprehensive facilities for software development, including:

- **Editor**: For writing code.
- **Debugger**: For finding and fixing issues.
- **Build Automation**: To streamline development tasks.

Popular Python IDEs include:

- PyCharm
- VSCode
- Cloud9 (AWS-based)
- Eclipse

## Python File Structure

Python files have the `.py` extension. The entry point for execution in a Python file is found in [`main.py`](https://github.com/ndomah/1.-The-Basics/blob/main/2.%20Introduction%20to%20Python/files/main.py):

```python
if __name__ ==  "__main__":
    print("Welcome to Python")
```
**Output:**
```
Welcome to Python
```

## Core Python Concepts

### Math Expressions
Reference: [`math_exp.py`](https://github.com/ndomah/1.-The-Basics/blob/main/2.%20Introduction%20to%20Python/files/math_exp.py)

```python
x = 40
y = 15
print('sum: ', x + y)
print('diff: ', x - y)
print('product: ', x * y)
print('div: ', x / y)
print('quotient: ', x // y)
print('remainder: ', x % y)
print('power: ', x ** y)
```
**Output:**
```
sum: 55
diff: 25
product: 600
div: 2.6666666666666665
quotient: 2
remainder: 10
power: 1073741824000000000000000
```

### Strings
Reference: [`strings.py`](https://github.com/ndomah/1.-The-Basics/blob/main/2.%20Introduction%20to%20Python/files/strings.py) and [`strings_demo.py`](https://github.com/ndomah/1.-The-Basics/blob/main/2.%20Introduction%20to%20Python/files/strings_demo.py)

```python
print('Hello' + ' ' + 'World')
print('Hello' * 3)
print(str(1))
s = 'hi'
print(s[0])
```
**Output:**
```
Hello World
HelloHelloHello
1
h
```

### Variables
Reference: [`variables.py`](https://github.com/ndomah/1.-The-Basics/blob/main/2.%20Introduction%20to%20Python/files/variables.py)

```python
x = 1
y = False
s = 'hello'
x = 1000
y = x
```
**Output:**
```
type of x : <class 'int'>,  id of x : 9793088
type of y : <class 'bool'>, id of y : 9478112
type of s : <class 'str'>,  id of s : 140661482079792
```

### Indexing and Slicing
Reference: [`indexing_and_slicing.py`](https://github.com/ndomah/1.-The-Basics/blob/main/2.%20Introduction%20to%20Python/files/indexing_and_slicing.py)

```python
careers = ['ml', 'de', 'ds', 'ui', 'api']
print(careers[0])
print(careers[-1])
print(careers[:3])
print(careers[-3:])
```
**Output:**
```
ml
api
['ml', 'de', 'ds']
['ds', 'ui', 'api']
```

### Loops
Reference: [`loops.py`](https://github.com/ndomah/1.-The-Basics/blob/main/2.%20Introduction%20to%20Python/files/loops.py)

```python
for x in 'python':
    print(x)
```
**Output:**
```
p
y
t
h
o
n
```

### Functions
Reference: [`functions.py`](https://github.com/ndomah/1.-The-Basics/blob/main/2.%20Introduction%20to%20Python/files/functions.py)

```python
def sum_two(a, b=0):
    return a + b

def sum_three(a, b, c):
    return a + b + c

print(sum_two(1))
print(sum_three(1, 2, 3))
```
**Output:**
```
2 Sum : 1
3 Sum : 6
```

### Reading/Writing JSON and CSV
Reference: [`reading_json.py`](https://github.com/ndomah/1.-The-Basics/blob/main/2.%20Introduction%20to%20Python/files/reading_json.py) and [`csv_read.py`](https://github.com/ndomah/1.-The-Basics/blob/main/2.%20Introduction%20to%20Python/files/csv_read.py)

```python
import json
import csv

data = json.loads('{"name": "Jack", "age": 29}')
with open('emp_file.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file)
    employee_writer.writerow(['Name', 'Department'])
```
**Output:**
```
File exists : True
Processed 2 lines.
```

## Running Python Code

You can start a Python console within the terminal to perform direct coding and test scripts interactively.

To execute a Python script, use the command:

```bash
python filename.py
```

## Additional Resources

- [Official Python Documentation](https://docs.python.org/3/)
- [Python Tutorial by W3Schools](https://www.w3schools.com/python/)
- [Python for Beginners - Microsoft](https://learn.microsoft.com/en-us/training/paths/python-for-beginners/)
