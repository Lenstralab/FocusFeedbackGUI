from time import sleep
from parfor import parfor
from timeit import default_timer as time

def my_fun(*args, **kwargs):
    tic =time()
    @parfor(range(10), (3,), nP=10)
    def fun(i, a):
        sleep(1)
        return a*i**2
    toc = time()
    print(toc-tic)
    return fun

if __name__ == '__main__':
    my_fun()