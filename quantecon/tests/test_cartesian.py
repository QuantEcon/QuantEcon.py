"""
Author: Pablo Winant
Filename: test_cartesian.py

Tests for cartesian.py file

"""

from quantecon.cartesian import cartesian, _repeat_1d

def test_cartesian_C_order():

    from numpy import linspace
    x = linspace(0,9,10)

    prod = cartesian([x,x,x])

    correct = True
    for i in range(999):
        n = prod[i,0]*100+prod[i,1]*10+prod[i,2]
        correct *= (i == n)

    assert(correct)

def test_cartesian_C_order_int_float():

    from numpy import arange, linspace

    x_int = arange(10)
    x_float = linspace(0,9,10)
    prod_int = cartesian([x_int]*3)
    prod_float = cartesian([x_float]*3)
    assert(prod_int.dtype==x_int.dtype)
    assert(prod_float.dtype==x_float.dtype)
    assert( abs(prod_int-prod_float).max()==0)

def test_cartesian_F_order():

    from numpy import linspace
    x = linspace(0,9,10)

    prod = cartesian([x,x,x], order='F')

    correct = True
    for i in range(999):
        n = prod[i,2]*100+prod[i,1]*10+prod[i,0]
        correct *= (i == n)

    assert(correct)

def test_performance_C():

    from numpy import linspace, column_stack, repeat, tile
    import time

    N_x = 1000
    N_y = 7777
    x = linspace(1,N_x,N_x)
    y = linspace(1,N_y,N_y)

    cartesian([x[:10],y[:10]]) # warmup

    t1 = time.time()
    for i in range(100):
        prod = cartesian([x,y])
    t2 = time.time()
    # print(prod.shape)

    # compute the same produce using numpy:
    import numpy

    t3 = time.time()
    for i in range(100):
        prod_numpy = column_stack([
            repeat(x,N_y),
            tile(y,N_x)
        ])
    t4 = time.time()

    print("Timings for 'cartesian' (C order)")
    print("Cartesian: {}".format(t2-t1))
    print("Numpy:     {}".format(t4-t3))
    assert(abs(prod-prod_numpy).max()==0)

def test_performance_F():

    from numpy import linspace, column_stack, repeat, tile
    import time

    N_x = 1000
    N_y = 7777
    x = linspace(1,N_x,N_x)
    y = linspace(1,N_y,N_y)

    cartesian([x[:10],y[:10]]) # warmup

    t1 = time.time()
    for i in range(100):
        prod = cartesian([x,y], order='F')
    t2 = time.time()
    # print(prod.shape)

    # compute the same produce using numpy:
    import numpy

    t3 = time.time()
    for i in range(100):
        prod_numpy = column_stack([
            tile(x,N_y),
            repeat(y,N_x)
        ])
    t4 = time.time()

    print("Timings for 'cartesian'(Fortran order)")
    print("Cartesian: {}".format(t2-t1))
    print("Numpy:     {}".format(t4-t3))
    assert(abs(prod-prod_numpy).max()==0)

def test_mlinsplace():

    from numpy import linspace
    from quantecon.cartesian import mlinspace

    grid1 = mlinspace([-1,-1],[2,3],[30,50])
    grid2 = cartesian([linspace(-1,2,30), linspace(-1,3,50)])

def test_tile():

    from numpy import linspace, tile, zeros
    x = linspace(1,100, 100)

    import time
    t1 = time.time()
    t_repeat = zeros(100*1000)
    _repeat_1d(x,1,t_repeat)
    t2 = time.time()

    t3 = time.time()
    t_numpy = tile(x, 1000)
    t4 = time.time()

    print("Timings for 'tile' operation")
    print("Repeat_1d: {}".format(t2-t1))
    print("Numpy:     {}".format(t4-t3))

    assert( abs(t_numpy-t_repeat).max())

def test_repeat():

    from numpy import linspace, repeat, zeros
    x = linspace(1,100  , 100)

    import time
    t1 = time.time()
    t_repeat = zeros(100*1000)
    _repeat_1d(x,1000,t_repeat)
    t2 = time.time()

    t3 = time.time()
    t_numpy = repeat(x, 1000)
    t4 = time.time()

    print("Timings for 'repeat' operation")
    print("Repeat_1d: {}".format(t2-t1))
    print("Numpy:     {}".format(t4-t3))

    assert( abs(t_numpy-t_repeat).max())




if __name__ == '__main__':
    test_cartesian_C_order()
    test_cartesian_F_order()
    test_performance_C()
    test_performance_F()
    test_tile()
    test_repeat()
