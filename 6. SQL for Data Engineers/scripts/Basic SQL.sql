-- Basic SQL
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