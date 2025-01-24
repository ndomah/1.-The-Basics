# Python for Data Engineers

## Advanced Python Concepts

### Classes and Modules

Python allows creating classes with attributes and functions and using modules to import and instantiate them.

#### Classes
- Create a class definition with attributes and functions.
- Instantiate the class and call its methods.

Refer to `my_classes.py` for implementation.

```python
from my_classes import MyAge

my_age = MyAge("1980-01-01", "Mr James")
print(my_age.show_me_my_age())
```
**Output:**
```
Mr James, you are so young, only 42 years old!
```

#### Modules
- Import custom class modules.
- Instantiate and call functions from the imported modules.

Refer to `modules.py` for details.

```python
from my_classes import MyAge

my_age = MyAge("1980-01-01", "Mr James")
my_age.show_me_my_age()
```
**Output:**
```
Mr James, you are so young, only 42 years old!
```

### Exception Handling

Exception handling in Python helps catch and raise errors effectively.

Refer to `exception_handling.py` for implementation details.

```python
try:
    e_commerce_csv_df = pd.read_csv('fake_data.csv')
except FileNotFoundError as error:
    print(f"{error}, please provide a correct path to the file!")
```
**Output:**
```
[Errno 2] No such file or directory: 'fake_data.csv', please provide a correct path to the file!
```

### Logging

Logging is used to track events that happen while running a program.

Refer to `my_logging.py` and check `reading_csvs.log` for practical usage.

```python
import logging
logging.basicConfig(filename="reading_csvs.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("10 is positive?")
```
**Example log output:**
```
2024-12-07 14:04:02,032 - INFO - 10 is positive?
```

## Data Engineering Concepts

### Datetime

Datetime operations include:

Refer to `my_datetime.py` for implementation.

```python
from datetime import datetime

now = datetime.now()
print(now)
```
**Output:**
```
2024-12-07 22:37:47.048226
```

### JSON

JSON is widely used in logging and APIs for data exchange.

Refer to `my_json.py` for examples.

```python
import json
with open("data_subset.json") as json_file:
    data = json.load(json_file)
print(data[0])
```
**Output:**
```
{'InvoiceNo': 536370, 'StockCode': 22492, 'Description': 'MINI PAINT SET VINTAGE', 'Quantity': 36, 'InvoiceDate': '12/1/2010 8:45', 'UnitPrice': 0.65, 'CustomerID': 12583, 'Country': 'France'}
```

### JSON Validation

Ensuring JSON data is in the correct structure.

Refer to `json_validation.py` and `test_json_validation.py` for implementation.

```python
pytest
```
**Output:**
```
==== 4 passed in 0.11s ====
```

### Requests (APIs)

Refer to `my_requests.py` for implementation.

```python
import requests
response = requests.get("https://api.disneyapi.dev/character")
print(response.status_code)
```
**Output:**
```
200
```

### Pandas & NumPy

Python libraries such as Pandas and NumPy are essential for data manipulation.

Refer to `my_pandas.py` and `my_numpy.py` for implementation details.

```python
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")
print(np.mean(data['Quantity'].to_numpy()))
```
**Output:**
```
12.785
```

### Working with Databases

Using PostgreSQL with Python:

Refer to `my_psycopg2.py` for database integration.

```python
import psycopg2
conn = psycopg2.connect(dbname="python_course", user="postgres", password="example", host="localhost")
cur = conn.cursor()
cur.execute("SELECT * FROM transactions WHERE InvoiceNo = '536370'")
records = cur.fetchall()
print(records)
```

## Running Python Code

To execute Python scripts, use the following command:

```bash
python filename.py
```

## Additional Resources

- [Python Official Documentation](https://docs.python.org/3/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

