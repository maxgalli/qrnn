import numpy as np
import pandas as pd
from qrnn import QRNN

from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster



def main():
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
      
    inputfile = 'df_data_EB_train.h5'
    n_evt = 2000000
    half_evts = int(n_evt/2)

    #load dataframe
    df_total = pd.read_hdf(inputfile)
    df_smp = df_total.sample(n_evt, random_state=100).reset_index(drop=True)

    df_train = df_smp[:half_evts] 
    df_test_raw  = df_smp[half_evts:] 

    #set features and target
    features = kinrho 
    target = variables[0]

    pTs = [25., 30., 35., 40., 45., 50., 60., 150.]

    #qs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    qs = [0.01, 0.05, 0.1]
    num_hidden_layers = 3
    num_units = [1000, 500, 100]
    act = ['tanh', 'softplus', 'elu']

    #for target in variables: 
        
    scale_para_file = 'scale_para_{}_{}.h5'.format(inputfile[3:-12], target)

    print('>>>>>>>>> train for variable {}'.format(target))
    qrnn = QRNN(df_train, features, target, scale_file=scale_para_file)


    # setup cluster 
    cluster = SLURMCluster(
            cores=1,
            memory="10GB",
            )
    client = Client(cluster)
    cluster.scale(len(qs))
    client.wait_for_workers(1)
    print("Waiting for at least one worker...")

    futures = [client.submit(
        qrnn.trainQuantile,
        q,
        num_hidden_layers,
        num_units,
        act,
        batch_size = 8192,
        save_file = 'model_{}_{}'.format(target,str(q).replace('.','p')),
        ) for q in qs ]

    results = client.gather(futures)
    print(results)


if __name__ == "__main__":
    main()
