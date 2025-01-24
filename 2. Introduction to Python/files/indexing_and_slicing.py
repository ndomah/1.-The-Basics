if __name__ == '__main__':
    careers = ['ml', 'de', 'ds', 'ui', 'api']

    #indexing
    for idx, career in enumerate(careers):
        print (idx, career)

    print (f'first : {careers[0]}, second : {careers[1]}')
    print (f'last : {careers[-1]}, second_last : {careers[-2]}')
    print (len(careers))
    # print (careers[len(careers)])

    # slicing
    print (f'First 3 items - {careers[:3]}')
    print (f'Last 3 items - {careers[-3:]}')
    print (f'Every other item - {careers[::2]}')
    print (f'All elements - {careers[:]}')

    print()

    a_dict = {'one': 1, 'two': 2, 'three': 3}
    print (a_dict)
    for key, value in a_dict.items():
        print (f'key : {key}, val : {value}')