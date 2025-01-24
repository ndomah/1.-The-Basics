if __name__ == '__main__':
    s = 'Hello'
    
    assert s.lower() == 'hello' # lower case
    assert s.upper() == 'HELLO' # upper case
    assert s.isalpha() # s is alphabetic
    assert s.isnumeric() == False # s is not numeric
    
    s = str(1234) # s -> '1234'
    assert s.isnumeric()
    s = 'Hello123'
    assert s.isalnum() 
    
    s = "10.5#11.73#19#10#100"
    split_str = s.split('#') # splits s on '#'
    print('split string: ', split_str)
    
    join_str = " ".join(split_str) # joins s with ' '
    print('joined string: ', join_str)
    
    replace_str = join_str.replace(' ', '-') # replaces ' ' with '-'
    print('replaced string: ', replace_str)
    print('string length: ', len(s)) # length of string   