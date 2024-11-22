
from Dataset import *
from Model import *

def check_label_polygon(label: LabelP):
    print(label.path_data)

def filer_polygon(label: LabelP):
    if not label.get():
        return False
    
    for label_name, polygon in label.get().items():
        if len(polygon) < 4:
            print("error", label_name, polygon, label.path_data, label.name_file)
            return False
    return True

def inpuct_polygon(label: LabelP):

    return label

    for label_name, polygon in label.get().items():
        new_polygon = []
        for i in range(0, len(polygon), 2):
            new_polygon.append([polygon[i], polygon[i+1]])

        label.set_polygon(label_name, new_polygon)

    return label


if __name__ == "__main__":
    path_dataset = "Test/test_Polygons"
    path_dataset = "dataset_new"

    labels = LabelsPolygon(path_dataset, filter=filer_polygon, map=inpuct_polygon, round=True)
    dataset = DatasetImage(path_dataset, labels, extension=".png", rotate=True, desired_size=(500, 500, 3))
    handler = HandlerDataset(dataset)
    
    for i in range(len(handler)):
        data = handler[i]
        labelP: LabelP = data[1]
        data: Image = data[0]
        print(f"{i}/{len(handler)}")
        # print(labelP)

        # for label, polygon in labelP.items():
        #     print(label)
        #     print(polygon)
        #     dataset.show_img(data, polygon, polygon=True)