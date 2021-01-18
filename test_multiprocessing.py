import time
import threading
import multiprocessing
def job(x):
    x = x * x
    time.sleep(5)
    if x == 0:
        'zero'
    elif x%2 == 0:
        'even'
    else:
        'odd'
    return

def runBaseLine():
    starttime = time.time()
    for i in range(0,10):
        job(i)
    print('Baseline took {} seconds'.format(time.time() - starttime))

def runMultithread():
    threads = list()
    starttime = time.time()
    for element in range(0,4):
        threads.append(threading.Thread(target=job, args=(element,)))
    for thread in threads:    
        thread.start()
    for thread in threads:
        thread.join()
    print('Multithread took {} seconds'.format(time.time() - starttime))

def runMultiprocessing():
    starttime = time.time()
    processes = list()
    for i in range(0,10):
        p = multiprocessing.Process(target=job, args=(i,))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    print('Multiprocessing took {} seconds'.format(time.time() - starttime))

def runMultiprocessingUsingPool():
    test = list(range(0,10))
    starttime = time.time()
    with multiprocessing.Pool() as p:
        p.map(job, test)
    print('Multiprocessing using pool took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":
    # runMultithread()
    # runMultiprocessing()
    runMultiprocessingUsingPool()