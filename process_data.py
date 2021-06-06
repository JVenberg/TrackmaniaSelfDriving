import os
import shutil
import csv
import random
from PIL import Image, ImageOps


IN_DIR = "raw_data"
OUT_DIR = "data"

IN_DATA_FILE = "telemetry.csv"
OUT_DATA_FILE = "labels.csv"
OUT_TRAIN_FILE = "train.csv"
OUT_TEST_FILE = "test.csv"

TRAIN_TEST_SPLIT = 0.8


def write_data_row(writer, img, file_name, speed, steering):
    writer.writerow({"img_file": file_name, "speed": speed, "steering": steering})
    img.save(os.path.join(OUT_DIR, file_name))

def convert_raw_to_output():
    shutil.rmtree(OUT_DIR)
    os.mkdir(OUT_DIR)
    with open(os.path.join(IN_DIR, IN_DATA_FILE), newline="") as csv_in_file, open(
        os.path.join(OUT_DIR, OUT_DATA_FILE), "w", newline=""
    ) as csv_out_file:
        reader = csv.DictReader(csv_in_file)
        fieldnames = ["img_file", "speed", "steering"]
        writer = csv.DictWriter(csv_out_file, fieldnames=fieldnames)
        writer.writeheader()

        rows = list(reader)
        for i, row in enumerate(rows):
            img = Image.open(os.path.join(IN_DIR, row["img_file"]))

            write_data_row(writer, img, row["img_file"], row["speed"], row["steering"])
            write_data_row(
                writer,
                ImageOps.mirror(img),
                os.path.splitext(row["img_file"])[0] + "_flip.png",
                row["speed"],
                -float(row["steering"]),
            )
            if i % 100 == 0:
                print(str((i + 1) / len(rows) * 100) + '%')

def split_data():
    with open(os.path.join(OUT_DIR, OUT_DATA_FILE), newline="") as csv_label_file, \
        open(os.path.join(OUT_DIR, OUT_TRAIN_FILE), 'w', newline="") as csv_train_file, \
        open(os.path.join(OUT_DIR, OUT_TEST_FILE), 'w', newline="") as csv_test_file:
        reader = csv.DictReader(csv_label_file)
        train_writer = csv.DictWriter(csv_train_file, fieldnames=reader.fieldnames)
        test_writer = csv.DictWriter(csv_test_file, fieldnames=reader.fieldnames)
        train_writer.writeheader()
        test_writer.writeheader()

        rows = list(reader)
        random.shuffle(rows)
        train = rows[:int((len(rows)) * TRAIN_TEST_SPLIT)]
        test = rows[int((len(rows)) * TRAIN_TEST_SPLIT):]

        train_writer.writerows(train)
        test_writer.writerows(test)

convert_raw_to_output()
split_data()