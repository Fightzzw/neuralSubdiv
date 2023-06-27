import json
import os

def main():

    data = {
        "train_pkl": "./data_PKL/10k_surface_fr0.06_ns3_nm5550_test/test_total_564.pkl",
        "valid_pkl": "./data_PKL/10k_surface_fr0.06_ns3_nm5550_valid/valid_total_563.pkl",
        "output_path": './jobs/10k_surface_fr0.06_ns3_nm5550_test+valid/',
        "epochs": 3000,
        "lr": 2e-3, 
        "device": 'cuda',
        "Din": 6,
        "Dout": 128,
        "h_initNet": [128, 128,128, 128,128, 128],
        #"h_edgeNet":  [128, 128],
        #"h_vertexNet": [128, 128],
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