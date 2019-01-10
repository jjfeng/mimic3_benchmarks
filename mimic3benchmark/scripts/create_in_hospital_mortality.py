import sys
import os
import argparse
import pandas as pd
import random
import numpy as np


def parse_args(args):
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument(
            'root_path',
            type=str,
            help="Path to root folder containing train and test sets.")
    parser.add_argument(
            '--output-path',
            type=str,
            default="../../data/mimic/in-hospital-mortality/",
            help="Directory where the created data should be stored.")
    parser.add_argument(
            '--seed',
            type=int,
            default=100,
            help="random seed")
    parser.add_argument(
            '--train-csv',
            type=str,
            default="../../data/mimic/train_ids.csv",
            help="csv file with input train ids")
    parser.add_argument(
            '--test-csv',
            type=str,
            default="../../data/mimic/test_ids.csv",
            help="csv file with input test ids")
    args, _ = parser.parse_known_args()
    assert args.output_path != args.root_path
    args.patient_id_csvs = {
            "train": args.train_csv,
            "test": args.test_csv}
    return args

def process_partition(args, partition, eps=1e-6, n_hours=48):
    patient_id_csv = args.patient_id_csvs[partition]
    patients = np.array(np.genfromtxt(patient_id_csv, delimiter=","), dtype=int)
    print(partition, "patient ids", patients)

    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    list_file_data = []
    for (patient_index, patient) in enumerate(patients):
        patient = str(patient)
        patient_folder = os.path.join(args.root_path, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
                first_row_label = label_df.iloc[0]

                # empty label file
                if label_df.shape[0] == 0:
                    continue

                mortality = int(first_row_label["Mortality"])
                los = 24.0 * first_row_label['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                if los < n_hours - eps:
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)
                output_lb_filename = patient + "_" + lb_filename
                with open(os.path.join(output_dir, output_lb_filename), "w") as outfile:
                    outfile.write("Age,Gender,Ethnicity\n")
                    outfile.write("%f,%d,%d" % (first_row_label["Age"], first_row_label["Gender"], first_row_label["Ethnicity"]))

                list_file_data.append((patient, output_ts_filename, output_lb_filename, n_hours, mortality))

        if (patient_index + 1) % 100 == 0:
            print("processed {} / {} patients".format(patient_index + 1, len(patients)), end='\r')

    print("\n", len(list_file_data))
    list_file_data = sorted(list_file_data)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('patient,stay,meta,period_length,y_true\n')
        for list_file_entry in list_file_data:
            listfile.write('%s,%s,%s,%d,%d\n' % list_file_entry)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "train")
    process_partition(args, "test")


if __name__ == '__main__':
    main()
