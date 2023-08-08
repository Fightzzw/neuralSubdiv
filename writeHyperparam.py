import json
import os

def main():

    data = {
        "train_pkl": "./data_PKL/203289_sf_f1000_ns3_nm10.pkl",
        "valid_pkl": "./data_PKL/203289_sf_f1000_ns3_nm10.pkl",
        "output_path": './jobs/203289_sf_f1000_ns3_nm10/',
        "epochs": 3000,
        "batch_size": 1,
        "lr": 2e-3, 
        "device": 'cuda',
        "Din": 6,
        "Dout": 32,
        "h_initNet": [32, 32],
        "h_edgeNet":  [32, 32],
        "h_vertexNet": [32, 32],
        "numSubd": 3,
    }

    # create directory
    if not os.path.exists(data['output_path']):
        print('Creating Job directory: ' + data['output_path'])
        os.mkdir(data['output_path'])

    # write hyper parameters into a json file
    with open(data['output_path'] + 'hyperparameters.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()