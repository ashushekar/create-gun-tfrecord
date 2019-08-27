import cv2
import os

from json2xml import json2xml, readfromurl, readfromstring, readfromjson
import json

import os
import glob

# os.chdir(r"data/Images")
# for index, oldfile in enumerate(glob.glob("*.jpeg"), start=1):
#     newfile = 'gpl_{:08}.jpeg'.format(index)
#     os.rename (oldfile,newfile)

def get_xmls(id):
    id = id.split('/')[2]
    id = id.split('.')[0]
    id_str = "{num:08}".format(num=int(id))
    img = cv2.imread("data/Images/gpl_" + id_str + ".jpeg", 1)

    coordinates = []
    with open("data/Labels/" + id + ".txt") as f:
        # coordinates = f.readlines()[1].split()
        lines = f.readlines()
        for line_no in range(1, len(lines)):
            coordinates.append(lines[line_no].split())

    height, width, channels = img.shape

    json_input = {
        "annotation": {
            "folder": "YouTubeObjects",
            "filename": "gpl_" + id_str + ".jpeg",
            "source": {
                "database": "The YouTubeObjects Database 2015",
                "annotation": "YouTubeObjects 2015",
                "image": "flickr",
                "flickrid": "341012865"
            },
            "owner": {
                "flickrid": "Ashwin",
                "name": "VID training set"
            },
            "size": {
                "width": width,
                "height": height,
                "depth": channels
            },
            "segmented": 0
        }
    }
    object = []
    for coor in coordinates:
        object.append({
                "name": "gpl",
                "pose": "Frontal",
                "truncated": 0,
                "occluded": 0,
                "bndbox": {
                    "xmin": coor[0],
                    "ymin": coor[1],
                    "xmax": coor[2],
                    "ymax": coor[3],
                },
                "difficult": 0
            })
    json_input["annotation"]["object"] = object

    json_output = json.dumps(json_input)
    json_output = json.loads(json_output)

    myfile = open("data/output/gpl_" + id_str + ".xml", "w")
    myfile.write(json2xml.Json2xml(json_output["annotation"]).to_xml())


for _, labelfile in enumerate(glob.glob("data/Labels/*.txt"), start=1):
    get_xmls(labelfile)
