from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import random
random.seed(49297)


def process_partition(args, partition, eps=1e-6, n_hours=48):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    list_file_data = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.root_path, partition, patient)
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
    if partition == "train":
        random.shuffle(list_file_data)
    if partition == "test":
        list_file_data = sorted(list_file_data)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('patient,stay,meta,period_length,y_true\n')
        for list_file_entry in list_file_data:
            listfile.write('%d,%s,%s,%d,%d\n' % list_file_entry)


def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
