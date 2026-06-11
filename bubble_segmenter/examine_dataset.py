from config import IMG_SIZE
from utils import save_dataset
from data_load import get_tvt_split

train_ds, val_ds, test_ds = get_tvt_split("single_images", IMG_SIZE, no_mask=True)
ds = train_ds.concatenate(val_ds).concatenate(test_ds)
save_dataset(ds.take(20), "tmp_results", "result")
