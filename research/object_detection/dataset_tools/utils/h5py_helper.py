import numpy as np
import h5py


def write_to_h5py(data, data_name, file_name):
    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset(data_name,  data=data)


def read_from_h5py(data_name, file_name):
    with h5py.File(file_name, 'r') as hf:
        data = hf[data_name][:]
    return data

if __name__ == '__main__':
    from tempfile import TemporaryFile
    file = 'test.h5'
    data_to_write = np.random.random(size=(100,20)) # or some such
    write_to_h5py(data_to_write, 'test', file)
    data_from_file = read_from_h5py('test', file)
    print((data_to_write == data_from_file).all())