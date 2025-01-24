-- SELECT & Aggregation
-- querying Employee table for all records
SELECT * 
FROM Employee;

-- querying Employee table for attributes related to employee location
SELECT EmployeeID, LastName, FirstName, City, State, Country
FROM Employee;

-- How many customers do we have?
SELECT COUNT(*)
FROM Customer;

-- What is the total number of sales by customer id?
SELECT CustomerID, COUNT(*) AS NumSales
FROM Invoice
GROUP BY CustomerId;

-- What is the average total amount across all invoices?
SELECT AVG(Total) AS AverageInvoiceAm
FROM Invoice;

-- What is the average invoice total for each country where invoices were billed?
SELECT BillingCountry, AVG(Total) AS AverageInvoiceAm
FROM Invoice
GROUP BY BillingCountry;

-- What was the total amount invoiced in our media store?
SELECT SUM(Total) AS TotalRevenue
FROM Invoice;

-- What is the total number of sales and total revenue by customer ID
SELECT CustomerID, COUNT(*) AS NumSales, SUM(Total) AS Revenue
FROM Invoice
GROUP BY CustomerID;

-- What was the earliest transaction data
SELECT MIN(InvoiceDate) AS EarliestInvoiceDate
FROM Invoice;

/*
What is the total number of invoices, total revenue, average invoice amount,
smallest invoice amount, and largest invoice amount for each distinct
InvoiceDate and BillingCountry?
*/
SELECT InvoiceDate,
	   BillingCountry,
	   COUNT(*) AS TotalInvoices,
	   SUM(Total) AS TotalRevenue,
	   AVG(Total) AS AverageInvoiceAm,
	   MIN(Total) AS SmallestInvoiceAm,
	   MAX(Total) AS LargestInvoiceAm,
	   COUNT(DISTINCT CustomerID) AS NumCustomers
FROM Invoice
GROUP BY 1, 2;

/*
What countries had a total amount invoiced in a month greater than the average invoice 
across all billing countries in a given month?
*/
SELECT BillingCountry,
	   STRFTIME('%m', InvoiceDate) AS InvoiceMonth,
	   STRFTIME('%Y', InvoiceDate) AS InvoiceYear,
	   SUM(Total) AS TotalRevenue
FROM Invoice
WHERE STRFTIME('%Y', InvoiceDate) = '2013'
GROUP BY BillingCountry, STRFTIME('%m', InvoiceDate), STRFTIME('%Y', InvoiceDate)
HAVING SUM(Total) > (
	SELECT AVG(Total)
	FROM Invoice
	GROUP BY STRFTIME('%m', InvoiceDate), STRFTIME('%Y', InvoiceDate)
)
ORDER BY BillingCountry, InvoiceMonth, InvoiceYear;

-- Find the number of unique CustomerIDs (59)
SELECT COUNT(DISTINCT CustomerID)
FROM Customer;

-- Inserting recrods for testing
INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) VALUES (60, 'ProspectFName', 'ProspectLNAME', 'Germany', 'test@test.com');