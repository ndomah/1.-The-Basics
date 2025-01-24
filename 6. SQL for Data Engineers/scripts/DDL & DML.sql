-- Basic SQL: DDL & DML 
-- creating a Band table
CREATE TABLE Band (
ArtistID INT,
LeadSinger TEXT,
YearStart INT,
YearEnd INT
);

-- insert initial Band values
INSERT INTO Band VALUES (22, 'Robert Plant', 1968, 1980);

-- insert more values
INSERT INTO Band (ArtistId, LeadSinger, YearStart, YearEnd) VALUES
(50,'James Hetfield',1981,2024),
(51,'Freddie Mercury',1970,1991),
(52,'Paul Stanley and Gene Simmons',1973,2024),
(110,'Kurt Cobain',1987,1994),
(120,'David Gilmour and Roger Waters',1965,1995),
(127,'Anthony Kiedis',1983,2024),
(140,'Jim Morrison',1965,1971),
(179,'Klaus Meine',1965,2024),
(187,'Marcelo Camelo',1997,2007);

-- create an empty table copy
CREATE TABLE Band_copy(
ArtistID INT,
LeadSinger TEXT,
YearStart INT,
YearEnd INT
);

-- inserting values to the new table with SELECT using the existing table
INSERT INTO Band_copy (ArtistId, LeadSinger, YearStart, YearEnd)
SELECT *
FROM Band;

-- drop the original Band table
DROP table Band;

-- creating a new Band table with the NOT NULL constraint
CREATE TABLE Band(
ArtistId INT,
LeadSinger TEXT,
YearStart NOT NULL,
YearEnd INT
);

-- inserting a record that doesn't comply with the constraint YearStart cannot be null
INSERT INTO Band (ArtistId, LeadSinger, YearEnd) VALUES
(22, 'Robert Plant', 1980),
(50,'James Hetfield',2024),
(51,'Freddie Mercury',1991),
(52,'Paul Stanley and Gene Simmons',2024),
(110,'Kurt Cobain',1994),
(120,'David Gilmour and Roger Waters',1995),
(127,'Anthony Kiedis',2024),
(140,'Jim Morrison',1971),
(179,'Klaus Meine',2024),
(187,'Marcelo Camelo',2007);

-- dropping the table again
DROP TABLE Band;

-- creating an empty table with the unique constraint
CREATE TABLE Band (
ArtistId INT,
LeadSinger TEXT UNIQUE,
YearStart INT,
YearEnd INT
);

-- inserting a record that doesn't comply with the unique lead singer constraint
INSERT INTO Band (ArtistId, LeadSinger, YearStart, YearEnd) VALUES
(22, 'Robert Plant', 1968, 1980),
(50,'Jim Morrison',1981,2024),
(51,'Freddie Mercury',1970,1991),
(52,'Paul Stanley and Gene Simmons',1973,2024),
(110,'Kurt Cobain',1987,1994),
(120,'David Gilmour and Roger Waters',1965,1995),
(127,'Anthony Kiedis',1983,2024),
(140,'Jim Morrison',1965,1971),
(179,'Klaus Meine',1965,2024),
(187,'Marcelo Camelo',1997,2007);

-- dropping the table
DROP table Band;

-- creating an empty table with the PRIMARY KEY constraint
CREATE TABLE Band (
ArtistId INT PRIMARY KEY,
LeadSinger TEXT,
YearStart INT,
YearEnd INT
);

-- inserting a record that doesn't comply with the contraint id... should be unique and not null
INSERT INTO Band (ArtistId, LeadSinger, YearStart, YearEnd) VALUES
(22, 'Robert Plant', 1968, 1980),
(22, 'Robert Plant', 1968, 1980),
(50,'James Hetfield',1981,2024),
(51,'Freddie Mercury',1970,1991),
(52,'Paul Stanley and Gene Simmons',1973,2024),
(110,'Kurt Cobain',1987,1994),
(120,'David Gilmour and Roger Waters',1965,1995),
(127,'Anthony Kiedis',1983,2024),
(140,'Jim Morrison',1965,1971),
(179,'Klaus Meine',1965,2024),
(187,'Marcelo Camelo',1997,2007);

-- correcting the previous query
INSERT INTO Band (ArtistId, LeadSinger, YearStart, YearEnd) VALUES
(22, 'Robert Plant', 1968, 1980),
(50,'James Hetfield',1981,2024),
(51,'Freddie Mercury',1970,1991),
(52,'Paul Stanley and Gene Simmons',1973,2024),
(110,'Kurt Cobain',1987,1994),
(120,'David Gilmour and Roger Waters',1965,1995),
(127,'Anthony Kiedis',1983,2024),
(140,'Jim Morrison',1965,1971),
(179,'Klaus Meine',1965,2024),
(187,'Marcelo Camelo',1997,2007);

-- adding a new column to the existing table
ALTER TABLE Artist ADD COLUMN BandFlag TEXT;

-- updating the new column in the Artist table
UPDATE Artist SET BandFlag='1' WHERE ArtistId=1;
UPDATE Artist SET BandFlag='1' WHERE ArtistId=22;
UPDATE Artist SET BandFlag='1' WHERE ArtistId=50;
UPDATE Artist SET BandFlag='1' WHERE ArtistId=51;
UPDATE Artist SET BandFlag='1' WHERE ArtistId=52;
UPDATE Artist SET BandFlag='1' WHERE ArtistId=110;
UPDATE Artist SET BandFlag='1' WHERE ArtistId=120;
UPDATE Artist SET BandFlag='1' WHERE ArtistId=127;
UPDATE Artist SET BandFlag='1' WHERE ArtistId=140;
UPDATE Artist SET BandFlag='1' WHERE ArtistId=179;
UPDATE Artist SET BandFlag='1' WHERE ArtistId=187;

-- deleting selected records from the table
DELETE FROM Band
WHERE ArtistID = 22;

-- creating a view
CREATE VIEW customer_v AS
SELECT CustomerID,
	   FirstName || ' ' || c.LastName AS CustomerName,
	   Company,
	   Address,
	   City,
	   State,
	   Country,
	   PostalCode,
	   Phone,
	   Fax,
	   Email
FROM Customer AS c;