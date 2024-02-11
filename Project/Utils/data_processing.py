import numpy.random as rnd
import sklearn.preprocessing
import netCDF4
import torch

from utils import path, single_folder, atmospheric_folder, static_variable, single_variable, \
    atmospheric_variable, resolution, abbr
from utils import low_year_train, max_year_train, low_year_val_test, max_year_val_test
from utils import lev_indexes
from utils import printProgressBar


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PreProcessing:
    def __init__(self, train_set):
        self.scaler = sklearn.preprocessing.StandardScaler()
        for i in range(10000):
            year = rnd.randint(0, (max_year_train - low_year_train) + 1)
            #hour = rnd.randint(0, 8760)
            hour = rnd.randint(0, 8753)
            if i < 50:
                self.scaler.fit(train_set[0][0])
                self.scaler.fit(train_set[1][0])
                self.scaler.fit(train_set[2][0])
            self.scaler.fit(train_set[3][year][hour])
            self.scaler.fit(train_set[4][year][hour])
            self.scaler.fit(train_set[5][year][hour])
            self.scaler.fit(train_set[6][year][hour])
            for j in range(len(lev_indexes)):
                self.scaler.fit(train_set[7][year][j][hour])
                self.scaler.fit(train_set[8][year][j][hour])
                self.scaler.fit(train_set[9][year][j][hour])
                self.scaler.fit(train_set[10][year][j][hour])
                self.scaler.fit(train_set[11][year][j][hour])
                self.scaler.fit(train_set[12][year][j][hour])

    def transform(self, dataset):
        transformed_set = []
        for i in range(len(dataset)):
            transformed_set.append(self.scaler.transform(dataset[i]))
        return transformed_set

    def process(self, dataset):
        processed_data = []
        for i in range(len(dataset)):                                       # len(dataset) == 141
            processed_data.append(self.transform([dataset[i]]))
        processed_dataset = torch.FloatTensor(processed_data).to(device)

        return processed_dataset


def define_sets():              # Loads .nc files and turns them into lists
    print("Start sets definition")
    train_set, validation_set, test_set = [], [], []
    # Constants
    print("Constants")
    lsm, orography, lat2d = [], [], []
    nc = netCDF4.Dataset(f"{path}/{static_variable}.nc")
    for i in range(0, 3):
        data = nc[abbr[i]]
        data_np = data[:]
        if i == 0:
            lsm.append(data_np)
        if i == 1:
            orography.append(data_np)
        if i == 2:
            lat2d.append(data_np)
    # Define train_set
    print("Train Set Processing")
    train_tisr, train_t2m, train_u10, train_v10 = [], [], [], []
    train_z, train_u, train_v, train_t, train_q, train_r, = [], [], [], [], [], []
    j, l = 0, (max_year_train + 1 - low_year_train)
    printProgressBar(j, l, prefix='Progress:', suffix='Complete', length=50)
    for year in range(low_year_train, max_year_train + 1):
        for i in range(3, len(abbr)):
            if 2 < i < 7:
                nc = netCDF4.Dataset(f"{path}/{single_folder[i-3]}/{single_variable[i-3]}{year}{resolution}.nc")
                data = nc[abbr[i]]
                data_np = data[:]
                # Remove the last 24 hours if this year has 366 days
                if data_np.shape[0] == 8784:
                    data_np = data_np[:8760]
            if i == 3:
                train_tisr.append(data_np)
            if i == 4:
                train_t2m.append(data_np)
            if i == 5:
                train_u10.append(data_np)
            if i == 6:
                train_v10.append(data_np)

            if 6 < i < 13:
                nc = netCDF4.Dataset(f"{path}/{atmospheric_folder[i-7]}/{atmospheric_variable[i-7]}{year}{resolution}.nc")
                data = nc[abbr[i]]
                data_np = data[:]
                # Remove the last 24 hours if this year has 366 days
                if data_np.shape[0] == 8784:
                    data_np = data_np[:8760]
                level = []
                for lev in lev_indexes:
                    level.append(data_np[:, lev])
            if i == 7:
                train_z.append(level)
            if i == 8:
                train_u.append(level)
            if i == 9:
                train_v.append(level)
            if i == 10:
                train_t.append(level)
            if i == 11:
                train_q.append(level)
            if i == 12:
                train_r.append(level)
        j += 1
        printProgressBar(j, l, prefix='Progress:', suffix='Complete', length=50)

    # Define validation_set and test_set
    print("Validation and Test Sets Processing")
    val_tisr, val_t2m, val_u10, val_v10 = [], [], [], []
    val_z, val_u, val_v, val_t, val_q, val_r, = [], [], [], [], [], []
    test_tisr, test_t2m, test_u10, test_v10 = [], [], [], []
    test_z, test_u, test_v, test_t, test_q, test_r, = [], [], [], [], [], []
    j, l = 0, (max_year_val_test - low_year_val_test)
    printProgressBar(j, l, prefix='Progress:', suffix='Complete', length=50)
    for year in range(low_year_val_test, max_year_val_test):
        for i in range(3, len(abbr)):
            if 2 < i < 7:
                nc = netCDF4.Dataset(f"{path}/{single_folder[i-3]}/{single_variable[i-3]}{year}{resolution}.nc")
                data = nc[abbr[i]]
                data_np = data[:]
                # Remove the last 24 hours if this year has 366 days
                if data_np.shape[0] == 8784:
                    data_np = data_np[:8760]
            if i == 3:
                val_tisr.append(data_np[0:4379])
                test_tisr.append(data_np[4380:8760])
            if i == 4:
                val_t2m.append(data_np[0:4379])
                test_t2m.append(data_np[4380:8760])
            if i == 5:
                val_u10.append(data_np[0:4379])
                test_u10.append(data_np[4380:8760])
            if i == 6:
                val_v10.append(data_np[0:4379])
                test_v10.append(data_np[4380:8760])

            if 6 < i < 13:
                nc = netCDF4.Dataset(f"{path}/{atmospheric_folder[i-7]}/{atmospheric_variable[i-7]}{year}{resolution}.nc")
                data = nc[abbr[i]]
                data_np = data[:]
                # Remove the last 24 hours if this year has 366 days
                if data_np.shape[0] == 8784:
                    data_np = data_np[:8760]
                level_val, level_test = [], []
                for lev in lev_indexes:
                    level_val.append(data_np[0:4380, lev])
                    level_test.append(data_np[4380:8760, lev])
            if i == 7:
                val_z.append(level_val)
                test_z.append(level_test)
            if i == 8:
                val_u.append(level_val)
                test_u.append(level_test)
            if i == 9:
                val_v.append(level_val)
                test_v.append(level_test)
            if i == 10:
                val_t.append(level_val)
                test_t.append(level_test)
            if i == 11:
                val_q.append(level_val)
                test_q.append(level_test)
            if i == 12:
                val_r.append(level_val)
                test_r.append(level_test)
        j += 1
        printProgressBar(j, l, prefix='Progress:', suffix='Complete', length=50)

    train_set.append(lsm), validation_set.append(lsm), test_set.append(lsm)
    train_set.append(orography), validation_set.append(orography), test_set.append(orography)
    train_set.append(lat2d), validation_set.append(lat2d), test_set.append(lat2d)
    train_set.append(train_tisr), validation_set.append(val_tisr), test_set.append(test_tisr)
    train_set.append(train_t2m), validation_set.append(val_t2m), test_set.append(test_t2m)
    train_set.append(train_u10), validation_set.append(val_u10), test_set.append(test_u10)
    train_set.append(train_v10), validation_set.append(val_v10), test_set.append(test_v10)
    train_set.append(train_z), validation_set.append(val_z), test_set.append(test_z)
    train_set.append(train_u), validation_set.append(val_u), test_set.append(test_u)
    train_set.append(train_v), validation_set.append(val_v), test_set.append(test_v)
    train_set.append(train_t), validation_set.append(val_t), test_set.append(test_t)
    train_set.append(train_q), validation_set.append(val_q), test_set.append(test_q)
    train_set.append(train_r), validation_set.append(val_r), test_set.append(test_r)
    print("End set definition")

    return train_set, validation_set, test_set
