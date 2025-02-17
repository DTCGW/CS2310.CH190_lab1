import tabmini
import argparse
from pathlib import Path
from tabmini.estimators import XGBoost
from tabmini.estimators.RF import RandomForest
from tabmini.estimators import LightGBM
from tabmini.estimators.TabNet import TabNet
from tabmini.types import TabminiDataset
from tabmini.utils import find_almost_constant_columns

def parse_arguments():
    parser = argparse.ArgumentParser(description="TabMini-Classification Options.")
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 4, 8, 10], 
        help="Type of model (1: XGBoost, 2: LightGBM, 4: Random Forest, 8: TabR, 10: TabNet)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="result",
        help="Folder to save result."
    )
    return parser.parse_args()

def main(args):
    working_directory = Path.cwd() / args.save_dir
    working_directory.mkdir(parents=True, exist_ok=True)

    # load dataset
    dataset: TabminiDataset = tabmini.load_dataset()
    dataset_name_lst = list(dataset.keys())

    # process
    for dt_name in dataset_name_lst:
        X, y = dataset[dt_name]
        
        if 2 in y.values: 
            y = (y == 2).astype(int)
        num_records = len(X)
        if args.model == 1: 
            model = XGBoost(small_dataset=True)
        elif args.model == 4: 
            model = RandomForest(small_dataset= True)
        elif args.model == 2: 
            model = LightGBM(small_dataset= True)
        elif args.model == 10:

            model = TabNet(small_dataset = True)
            # remove_threshold = 0.95 if len(X.columns) < 80 else 0.9
            # X = find_almost_constant_columns(X, remove_threshold, working_directory, dt_name, num_records)
        
        model.fit(X, y)

        model.save_results(filename= working_directory / f"{dt_name}_{num_records}.csv")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)