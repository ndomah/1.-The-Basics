def sum_two(a, b=0):
    return a + b

def sum_three(a, b, c):
    return a + b + c

def sum_args(*args):
    s_args = 0
    for arg in args:
        s_args += arg
    return s_args

def concatenate_strings(**kwargs):
    ans = ""
    for k, v in kwargs.items():
        print (f'key : {k}, value : {v}')
        ans += v
    return ans

def cool_func():
    pass


if __name__ == '__main__':
    print (f'2 Sum : {sum_two(1)}')
    print (f'3 Sum : {sum_three(1, 2, 3)}')
    print (f'4 Sum : {sum_args(1, 2, 3, 4)}')
    print (f'5 Sum : {sum_args(1, 2, 3, 4, 5)}') 
    print (concatenate_strings(one = 'python', two =  ' is ', three = 'fun'))