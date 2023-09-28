import argparse
import os
import traceback
import uuid

import pandas as pd
import numpy as np

# we want too encode special days as one variable and thus need to first reverse one-hot encoding
SPECIAL_DAYS = [
  "easter_day",
  "good_friday",
  "new_year",
  "christmas",
  "labor_day",
  "independence_day",
  "revolution_day_memorial",
  "regional_games",
  "fifa_u_17_world_cup",
  "football_gold_cup",
  "beer_capital",
  "music_fest",
]


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("--input-data-dir", type=str, default="/opt/ml/processing/input")
  parser.add_argument("--output-data-dir", type=str, default="/opt/ml/processing/output")

  args, _ = parser.parse_known_args()

  input_dir = args.input_data_dir
  input_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.parquet.snappy')]
  if len(input_files) == 0:
    raise RuntimeError()

  print(input_files) # debug
  raw_data = [pd.read_parquet(file) for file in input_files]
  data = pd.concat(raw_data)

  data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
  data["time_idx"] -= data["time_idx"].min()

  # add additional features
  data["month"] = data.date.dt.month.astype(str).astype("category") # categories have be strings
  data["log_volume"] = np.log(data.volume + 1e-8)
  data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
  data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")

  data[SPECIAL_DAYS] = data[SPECIAL_DAYS].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")

  output_dir = args.output_data_dir
  try:
    os.makedirs(output_dir, exist_ok=True)
    print("Successfully created directories")
  except Exception as _:
    # if the Processing call already creates these directories (or directory otherwise cannot be created)
    print("Could not make directories")
    traceback.print_exc()

  try:
    outfile_name = f"data-{str(uuid.uuid1()).split('-')[0]}.parquet.snappy"
    data.to_parquet(os.path.join(output_dir, outfile_name), index=False)
    print("Wrote files successfully")
  except Exception as e:
    print("Failed to write the files")
    traceback.print_exc()

  print("Completed running the processing job")

if __name__ == '__main__':
  main()
