# SQL for Data Engineers

## Introduction

This guide covers the fundamental concepts and advanced techniques of SQL for data engineers. Topics include:

- Introduction to databases and SQL
- Setting up SQLite & DBeaver with the Chinook database
- Basics of working with databases and data
- Data Definition Language (DDL)
- Data Manipulation Language (DML)
- Grouping and aggregation
- Joins
- Transaction Control Language (TCL)
- Common Table Expressions (CTE) and subqueries
- Database analysis with window functions
- Query optimization

## Database Management Systems & SQL

A **database** is a collection of stored data typically organized in tables, rows, and columns, following definitions from DAMA International.

A **database management system (DBMS)** is software that provides an interface for managing databases and interacting with stored data.

![Chinook Database Schema](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/img/fig2.png)

Refer to [`Chinook_Sqlite.sql`](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/scripts/Chinook_Sqlite.sql) for the database setup.

```sql
-- Example from Chinook database
SELECT * FROM Artist WHERE ArtistId = 1;
```

## Setting Up SQLite & DBeaver

### SQLite Overview

SQLite is a lightweight, serverless, self-contained database engine widely used in embedded systems and local applications.

![SQLite Overview](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/img/fig3.png)

#### Key Features of SQLite

- Embedded/Serverless
- Transactional database
- Zero configuration
- Cross-platform
- Compact size

## Data Types in SQLite

SQLite provides the following storage classes:

- `NULL`: Represents missing values
- `INTEGER`: Stores whole numbers
- `REAL`: Stores floating-point numbers
- `TEXT`: Stores text strings
- `BLOB`: Stores binary data

![SQLite Data Types](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/img/fig4.png)

### Data Type Comparison Across DBMS

![Data Type Comparison](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/img/fig5.png)

## Basic SQL Concepts

### SQL Commands Overview

![SQL Commands Overview](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/img/fig1.jpg)

### Data Definition Language (DDL)

DDL commands define the structure of a database. Examples include:

Refer to [`DDL & DML.sql`](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/scripts/DDL%20%26%20DML.sql) for more details.

```sql
CREATE TABLE Customers (
    CustomerId INTEGER PRIMARY KEY,
    FirstName TEXT NOT NULL,
    LastName TEXT NOT NULL
);
```

![DDL & DML](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/img/fig6.png)

### Data Manipulation Language (DML)

DML commands manipulate data within tables. Examples include:

```sql
INSERT INTO Customers (CustomerId, FirstName, LastName)
VALUES (1, 'John', 'Doe');

UPDATE Customers
SET FirstName = 'Jane'
WHERE CustomerId = 1;

DELETE FROM Customers
WHERE CustomerId = 1;
```

## SQL Query Execution Order

![Query Execution Order](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/img/fig7.png)

## Grouping and Aggregation

Refer to [`SELECT & Aggregation.sql`](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/scripts/SELECT%20%26%20Aggregation.sql) for aggregation queries.

```sql
SELECT Country, COUNT(*) AS TotalCustomers
FROM Customers
GROUP BY Country;
```

![Aggregation Functions](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/img/fig8.png)

### Joins

SQL Joins are used to combine rows from two or more tables based on a related column.

Types of joins:

- `INNER JOIN`: Returns records with matching values in both tables
- `LEFT JOIN`: Returns all records from the left table, and matched records from the right
- `RIGHT JOIN`: Returns all records from the right table, and matched records from the left
- `FULL OUTER JOIN`: Returns all records when there is a match in either table

![SQL Joins](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/img/fig9.png)

## Advanced SQL Concepts

### Transaction Control Language (TCL)

TCL manages transactions within databases and ensures data integrity through commands such as:

- `COMMIT`: Saves changes permanently
- `ROLLBACK`: Reverts changes to the last committed state
- `SAVEPOINT`: Sets intermediate save points

![ACID vs BASE](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/img/fig10.png)

### Common Table Expressions (CTE) and Subqueries

CTEs simplify complex queries and improve readability by defining temporary result sets.

![CTE vs Subquery](https://github.com/ndomah/1.-The-Basics/blob/main/6.%20SQL%20for%20Data%20Engineers/img/fig11.png)

```sql
WITH top_customers AS (
    SELECT CustomerId, COUNT(*) AS purchase_count
    FROM Invoice
    GROUP BY CustomerId
)
SELECT * FROM top_customers
WHERE purchase_count > 5;
```

### Window Functions

Window functions allow calculations across a set of table rows related to the current row.

Common window functions include:

- `ROW_NUMBER()`: Assigns a unique rank to rows
- `RANK()`: Assigns rank, allowing ties
- `DENSE_RANK()`: Similar to rank but without gaps
- `NTILE()`: Divides result set into a specified number of buckets
- `LAG()` and `LEAD()`: Access data from previous or next rows

#### Practical Examples:

```sql
SELECT CustomerId, InvoiceDate, Total,
       RANK() OVER (PARTITION BY CustomerId ORDER BY Total DESC) AS Rank
FROM Invoice;
```

```sql
SELECT CustomerId, InvoiceDate, Total,
       LAG(Total, 1, 0) OVER (PARTITION BY CustomerId ORDER BY InvoiceDate) AS PreviousTotal
FROM Invoice;
```

## Query Optimization

Optimizing SQL queries helps improve performance by:

- Creating proper indexes
- Avoiding SELECT * statements
- Using EXPLAIN PLAN to analyze queries
- Normalizing database schema

#### Practical Optimization Example:

```sql
CREATE INDEX idx_customer_id ON Invoice (CustomerId);
```

```sql
EXPLAIN QUERY PLAN
SELECT * FROM Invoice WHERE CustomerId = 5;
```

## Running SQL Queries

To run queries, use SQLite CLI or DBeaver with the Chinook database:

```sql
SELECT * FROM Album WHERE ArtistId = 1;
```

## Additional Resources

- [SQLite Official Documentation](https://sqlite.org/docs.html)
- [SQL Tutorial - W3Schools](https://www.w3schools.com/sql/)
- [Chinook Database Documentation](https://github.com/lerocha/chinook-database)
