from SGDRecommender import *
from BlockScheduler import *

if __name__ == '__main__':
    #Parse CLI arguments 
    if len(sys.argv) == 4:
        n_factors = int(sys.argv[1])
        alpha = float(sys.argv[2])
        beta1 = float(sys.argv[3])
        beta2 = float(sys.argv[4])
        MF_ALS = ExplicitMF(n_factors=n_factors, alpha=alpha, beta1=beta1, beta2=beta2)
    else:
        MF_ALS = ExplicitMF(n_factors=30)
    print(f"Using hyperparams: n_factors={MF_ALS.n_factors},alpha={MF_ALS.alpha},beta1={MF_ALS.beta1},beta2={MF_ALS.beta2}")
    
    #load data
    MF_ALS.load_samples_from_npy("./movielense_27.npy",1000000)
    #create grid of independent samples
    grid = MF_ALS.generate_indpendent_samples()
 
    # Create the Queue object
    m = multiprocessing.Manager()
    in_queue = m.Queue()
    out_queue = m.Queue()
    scheduler = BlockScheduler(grid,6)
    # Create a lock object to synchronize resource access
    lock = Lock()
 
    consumers = []
 
    # Create consumer processes
    for i in range(multiprocessing.cpu_count()):
        p = Process(target=consumer, args=(in_queue,out_queue, lock, MF_ALS))
        consumers.append(p)
        in_queue.put(scheduler.get_next())
 
    for c in consumers:
        c.start()
    
    #scheduler will handle putting and retreiving items from queues
    done = False
    while not done:
        updated_block = out_queue.get(timeout=20)
        next = scheduler.get_next(completed=updated_block)
        done = scheduler.check_completion()
        if done:
            print("Done!")
            for i in range(multiprocessing.cpu_count()):
                in_queue.put("000")
            break
        in_queue.put(next)

    
    # Like threading, we have a join() method that synchronizes our program
    for c in consumers:
        c.join()
    print(MF_ALS.test_grid)
    print('Parent process exiting...')