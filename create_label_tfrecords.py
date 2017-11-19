import argparse
import os
import tensorflow as tf
import glob

from object_detection.utils import dataset_util

parser = argparse.ArgumentParser()
parser.add_argument("images_path", help="path to the root folder that contains the images")
labelDict={}
labelCounter=1
IMAGE_WIDTH=720
IMAGE_HEIGHT=405
args = parser.parse_args()
images_path=args.images_path
tfrecords_filename = 'test_data_tfr.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
tfrecord_examples=[]

def create_label_tfrecord(images_path):
    '''
    Create label list and TFRecords for Object Detection API
    '''





    (_, folders, files) = next(os.walk(images_path))
    if "nologo" in folders:
        folders.remove("nologo")

    if len(folders) != 0:
        for subfolder in folders:
            sub_path = os.path.join(images_path, subfolder)

            create_label_tfrecord(sub_path)
    else:

        LABELS_FILE = "metadata.txt"
        #open metadata and load to memory
	global xmin,ymin,xmax,ymax,label,filename

        labels_file_path=os.path.join(images_path,LABELS_FILE)
        for filename in glob.glob(labels_file_path):
            with open(filename, 'rb') as f:
                metaDataLine= list(f)[-1]
            metavalues=metaDataLine.split(b',')
            label=metavalues[0]
            xmin=int(metavalues[1])
            ymin=int(metavalues[2])
            xmax=xmin+int(metavalues[3])
            ymax=ymin+int(metavalues[4])
        
 	if LABELS_FILE in files:
        	files.remove(LABELS_FILE)
        for file in files:
            image_data = tf.gfile.FastGFile(os.path.join(images_path,file), 'rb').read()
            image_format = 'jpeg'
            height = IMAGE_HEIGHT
            width = IMAGE_WIDTH
            xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
            xmins.append(float(xmin)/width)
            xmaxs = []  # List of normalized right x coordinates in bounding box
            # (1 per box)
            xmaxs.append(float(xmax)/width)
            ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
            ymins.append(float(ymin)/height)

            ymaxs = []  # List of normalized bottom y coordinates in bounding box
            ymaxs.append(float(ymax)/height)
            # (1 per box)
            classes_text = []  # List of string class name of bounding box (1 per box)
            classes_text.append(label)
            classes = []  # List of integer class id of bounding box (1 per box)
            classes.append(labelDict.get(label))

            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                b'image/filename': dataset_util.bytes_feature(filename),
                b'image/source_id': dataset_util.bytes_feature(filename),
                b'image/encoded': dataset_util.bytes_feature(image_data),
                b'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))

            tfrecord_examples.append(tf_example)

def create_label_dict(images_path):
    '''
    Create label list and TFRecords for Object Detection API
    '''
    global labelCounter


    (_, folders, files) = next(os.walk(images_path))
    if "nologo" in folders:
        folders.remove("nologo")

    if len(folders) != 0:
        for subfolder in folders:
            sub_path = os.path.join(images_path, subfolder)

            create_label_dict(sub_path)
    else:

        LABELS_FILE = "metadata.txt"

        #open metadata and load to memory

        labels_file_path=os.path.join(images_path,LABELS_FILE)
        for filename in glob.glob(labels_file_path):
            with open(filename, 'rb') as f:
                metaDataLine= list(f)[-1]
            metavalues=metaDataLine.split(b',')
            label=metavalues[0]
            if label not in labelDict:
                labelDict[label]=labelCounter
                labelCounter += 1


def serializeTFRecords(examples):
    for example in examples:
        writer.write(example.SerializeToString())

    writer.close()


def createpbtxt(dict):
    file=open('test_data_tfr.pbtxt','w')

    for key in dict:
        file.write("\n" +"item {" + "\n" +"id: "+str(dict[key])+"\n" +"name: "+"'"+key+"'"+"\n" +"}"+"\n")
    file.close()

create_label_dict(images_path)
create_label_tfrecord(images_path)
serializeTFRecords(tfrecord_examples)
createpbtxt(labelDict)
print(labelDict)

