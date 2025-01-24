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

Python files have the `.py` extension. The entry point for execution in a Python file is found in `main.py`:

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
Reference: `math_exp.py`

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
Reference: `strings.py` and `strings_demo.py`

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
Reference: `variables.py`

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

### Loops
Reference: `loops.py`

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

## Running Python Code

You can start a Python console within the terminal to perform direct coding and test scripts interactively.

To execute a Python script, use the command:

```bash
python filename.py
```
