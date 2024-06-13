from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PreProcessing:
    def __init__(self):
        self.scaler = sklearn.preprocessing.StandardScaler()

    def process(self, dataset):                                            # dataset has shape (batch_size, 141, 32, 64)
        processed_data = []
        for i in range(len(dataset)):                                      # len(dataset) == batch_size
            processed_batch = []
            for j in range(len(dataset[i])):                               # len(dataset[i]) == 141
                processed_batch.append(self.scaler.fit_transform(X=dataset[i][j]))
            processed_data.append(np.array(processed_batch, dtype=np.float32))
        processed_dataset = torch.FloatTensor(np.array(processed_data, dtype=np.float32)).to(device)
        return processed_dataset


def define_sets(task, val=0, test=0):              # Loads .nc files and turns them into lists
    # Constants
    if task == 'const':
        print("Constants Set Definition")
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
        constants_set = lsm + orography + lat2d
        return constants_set
    # Define train_set
    if task == 'train':
        print("Train Set Definition")
        train_tisr, train_t2m, train_u10, train_v10 = [], [], [], []
        train_z, train_u, train_v, train_t, train_q, train_r, = [], [], [], [], [], []
        j, l = 0, (max_bound_year_train - low_bound_year_train)
        printProgressBar(j, l, prefix='Progress:', suffix='Complete', length=50)
        for year in range(low_bound_year_train, max_bound_year_train):
            for i in range(3, len(abbr)):
                data, data_np = [], []
                if 2 < i < 7:
                    nc = netCDF4.Dataset(f"{path}/{single_folder[i - 3]}/{single_variable[i - 3]}{year}{resolution}.nc")
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

                level = []
                if 6 < i < 13:
                    nc = netCDF4.Dataset(f"{path}/{atmospheric_folder[i - 7]}/"
                                         f"{atmospheric_variable[i - 7]}{year}{resolution}.nc")
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
        train_list = []
        for i in range(max_year_train + 1):
            for j in range(8760):
                tr_list = []
                tr_list.append(train_tisr[i][j]), tr_list.append(train_t2m[i][j]), \
                tr_list.append(train_u10[i][j]), tr_list.append(train_v10[i][j])
                for lev in range(len(levels)):
                    tr_list.append(train_z[i][lev][j]), tr_list.append(train_u[i][lev][j]), \
                    tr_list.append(train_v[i][lev][j]), tr_list.append(train_t[i][lev][j]), \
                    tr_list.append(train_q[i][lev][j]), tr_list.append(train_r[i][lev][j])
                train_list.append(np.array(tr_list))
        train_set = train_list
        return train_set

    # Define validation_set and test_set
    if task == 'val_test':
        print("Validation and Test Sets Definition")
        val_tisr, val_t2m, val_u10, val_v10 = [], [], [], []
        val_z, val_u, val_v, val_t, val_q, val_r, = [], [], [], [], [], []
        test_tisr, test_t2m, test_u10, test_v10 = [], [], [], []
        test_z, test_u, test_v, test_t, test_q, test_r, = [], [], [], [], [], []
        j, l = 0, (max_bound_year_val_test - low_bound_year_val_test)
        printProgressBar(j, l, prefix='Progress:', suffix='Complete', length=50)
        for year in range(low_bound_year_val_test, max_bound_year_val_test):
            for i in range(3, len(abbr)):
                data, data_np = [], []
                if 2 < i < 7:
                    nc = netCDF4.Dataset(f"{path}/{single_folder[i - 3]}/{single_variable[i - 3]}{year}{resolution}.nc")
                    data = nc[abbr[i]]
                    data_np = data[0:4380]
                    # Remove the last 24 hours if this year has 366 days
                    if data_np.shape[0] == 8784:
                        data_np = data_np[:8760]
                if i == 3:
                  if val:
                    val_tisr.append(data_np[0:4380])
                  if test:
                    test_tisr.append(data_np[4380:8760])
                if i == 4:
                  if val:
                    val_t2m.append(data_np[0:4380])
                  if test:
                    test_t2m.append(data_np[4380:8760])
                if i == 5:
                  if val:
                    val_u10.append(data_np[0:4380])
                  if test:
                    test_u10.append(data_np[4380:8760])
                if i == 6:
                  if val:
                    val_v10.append(data_np[0:4380])
                  if test:
                    test_v10.append(data_np[4380:8760])

                level_val, level_test = [], []
                if 6 < i < 13:
                    nc = netCDF4.Dataset(f"{path}/{atmospheric_folder[i - 7]}/"
                                         f"{atmospheric_variable[i - 7]}{year}{resolution}.nc")
                    data = nc[abbr[i]]
                    data_np = data[0:4380]
                    # Remove the last 24 hours if this year has 366 days
                    if data_np.shape[0] == 8784:
                        data_np = data_np[:8760]
                    level_val, level_test = [], []
                    for lev in lev_indexes:
                      if val:
                        level_val.append(data_np[0:4380, lev])
                      if test:
                        level_test.append(data_np[4380:8760, lev])
                if i == 7:
                  if val:
                    val_z.append(level_val)
                  if test:
                    test_z.append(level_test)
                if i == 8:
                  if val:
                    val_u.append(level_val)
                  if test:
                    test_u.append(level_test)
                if i == 9:
                  if val:
                    val_v.append(level_val)
                  if test:
                    test_v.append(level_test)
                if i == 10:
                  if val:
                    val_t.append(level_val)
                  if test:
                    test_t.append(level_test)
                if i == 11:
                  if val:
                    val_q.append(level_val)
                  if test:
                    test_q.append(level_test)
                if i == 12:
                  if val:
                    val_r.append(level_val)
                  if test:
                    test_r.append(level_test)
            j += 1
            printProgressBar(j, l, prefix='Progress:', suffix='Complete', length=50)
        validation_list, test_list = [], []
        for i in range(max_year_val + 1):
            for j in range(4380):
                val_list, tst_list = [], []
                if val:
                  val_list.append(val_tisr[i][j]), val_list.append(val_t2m[i][j]), \
                  val_list.append(val_u10[i][j]), val_list.append(val_v10[i][j])
                if test:
                  tst_list.append(test_tisr[i][j]), tst_list.append(test_t2m[i][j]), \
                  tst_list.append(test_u10[i][j]), tst_list.append(test_v10[i][j])
                for lev in range(len(levels)):
                    if val:
                      val_list.append(val_z[i][lev][j]), val_list.append(val_u[i][lev][j]), \
                      val_list.append(val_v[i][lev][j]), val_list.append(val_t[i][lev][j]), \
                      val_list.append(val_q[i][lev][j]), val_list.append(val_r[i][lev][j])
                    if test:
                      tst_list.append(test_z[i][lev][j]), tst_list.append(test_u[i][lev][j]), \
                      tst_list.append(test_v[i][lev][j]), tst_list.append(test_t[i][lev][j]), \
                      tst_list.append(test_q[i][lev][j]), tst_list.append(test_r[i][lev][j])
                if val:
                    validation_list.append(np.array(val_list))
                if test:
                    test_list.append(np.array(tst_list))
        validation_set = validation_list
        test_set = test_list
        return validation_set, test_set
