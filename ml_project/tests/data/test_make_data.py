from ml_project.data.make_dataset import read_data, split_to_train_val
from ml_project.entities import SplittingParams


def test_load_dataset(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert len(data) > 10
    assert target_col in data.keys()


def test_split_dataset(dataset_path: str):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=123, val_size=val_size,)
    data = read_data(dataset_path)
    train, val = split_to_train_val(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10

