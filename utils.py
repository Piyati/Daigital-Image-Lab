import os
import cv2
import numpy as np


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
# Malisiewicz et al.
def non_max_suppression (boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes [:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom- -right y-coordinate of the bounding box
    area = (x2-x1+1)*(y2-y1+1)
    idxs = np.argsort (y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 -xx1 +1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],np.where (overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes [pick].astype("int")

# Build class for loading data
class DatasetLoad:
    # Initialize the calss
    def __init__(self,width = 64, height = 64, pre_type = 'Resize'):
        #give default preprocessing type is resize
        self.width = width
        self.height = height
        self.pre_type = pre_type
    # load dataset
    def load(self, pathes, verbose = -1):
        # verbose is used to show the loading status on screen
        # initial empty datas and labels
        datas = []
        labels = []
        # initial the path (main pathes, eg, datasets/animals
        mainfolders = os.listdir(pathes)
        # this os.listdir of that pathes will get everything in side the folder
        # eg, inside folder animals, it has cats, dogs and panda
        # then folders = [cats, dogs, panda]
        for folder in mainfolders:
            # loop inside mainfolders to read each folder
            # os.path.join use to join pathes with folder inside it
            # eg, pathes = datasets/animals
            # folder = cats
            # then fullpath will get datasets/animals/cats
            fullpath = os.path.join(pathes, folder)
            # list all files that has in fullpath
            # in this eg those file is image in each folder
            listfiles = os.listdir(fullpath)

            # Print on screen
            if verbose >0:
                print(' [INFO] loading ', folder, ' ...')

            for (i, imagefile) in enumerate(listfiles):
                # Define full path of image
                imagepath = pathes + '/' + folder + '/' + imagefile
                # Read image from imagepath
                image = cv2.imread(imagepath)
                # Give label image based on folder
                label = folder

                # Because of input image is not the same size
                # we have to resize it to have same size
                # That is why the default of pre-type is "Resize"

                if(self.pre_type == "Resize"):
                    image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                
                datas.append(image)
                labels.append(label)
                if verbose >0 and i>0 and (i+1)%verbose == 0:
                    print('[INFO] processed {}/{}'.format(i+1,len(listfiles)))
        return (np.array(datas), np.array(labels))