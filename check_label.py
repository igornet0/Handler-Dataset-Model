
from Dataset import *

def check_label_polygon(label: LabelP):
    print(label.get())

if __name__ == "__main__":
    labels = LabelsPolygon("dataset_new/3")
    for label in labels:
        check_label_polygon(label)