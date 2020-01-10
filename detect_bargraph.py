"""Uses trained neural network to predict graph types for each page of paper PDF
"""

import os.path
import argparse
import urllib

import tempfile
import subprocess
import glob
import itertools

from fastai.vision import *

try:
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors

    import skimage.io
    import matplotlib.pyplot as plt
    from colorspacious import cspace_convert
except:
    print('Calculations will fail if this is a worker')


def detect_graph_types_from_iiif(paper_id, pages, learner, debug=False):
    """Pull images from iiif server
    """

    print(paper_id, pages)

    url = "http://127.0.0.1:8182/iiif/2/biorxiv:{}.full.pdf/full/560,560/0/default.png?page={}"
    images = [open_image(io.BytesIO(requests.get(url.format(paper_id, pg)).content)) for pg in range(1, pages+1)]
    
    return detect_graph_types_from_list(images, learner)


def detect_graph_types_from_list(images, learner):
    """Predicts graph types for each image and returns pages with bar graphs
    """
    page_predictions = np.array([predict_graph_type(images[idx], learner) for idx in range(0, len(images))])
    bar_pages = np.where(page_predictions == 'bar')[0] + 1 #add 1 to page idx such that page counting starts at 1
    pie_pages = np.where(page_predictions == 'pie')[0] + 1
    hist_pages = np.where(page_predictions == 'hist')[0] + 1
    bardot_pages = np.where(page_predictions == 'bardot')[0] + 1
    box_pages = np.where(page_predictions == 'box')[0] + 1
    dot_pages = np.where(page_predictions == 'dot')[0] + 1
    violin_pages = np.where(page_predictions == 'violin')[0] + 1
    positive_pages = hist_pages.tolist() + bardot_pages.tolist() + box_pages.tolist() + dot_pages.tolist() + violin_pages.tolist()
    if len(positive_pages) > 0:
        positive_pages = list(set(positive_pages)) #remove duplicates and sort
        positive_pages.sort()

    class_pages = dict();  
    class_pages['bar'] = bar_pages.tolist()
    class_pages['pie'] = pie_pages.tolist()
    class_pages['hist'] = hist_pages.tolist()
    class_pages['bardot'] = bardot_pages.tolist()
    class_pages['box'] = box_pages.tolist()
    class_pages['dot'] = dot_pages.tolist()
    class_pages['violin'] = violin_pages.tolist()
    class_pages['positive'] = positive_pages
    

    return class_pages


def predict_graph_type(img, learner):
    """Use fastai model on each image to predict types of pages
    """
    class_names = {
        "0": ["approp"],
        "1": ["bar"],
        "2": ["bardot"],
        "3": ["box"],
        "4": ["dot"],
        "5": ["hist"],
        "6": ["other"],
        "7": ["pie"],
        "8": ["text"],
        "9": ["violin"]
    }
    
    pred_class,pred_idx,outputs = learner.predict(img)
    
    if pred_idx.sum().tolist() == 0: #if there is no predicted class 
        #(=no class over threshold) give out class with highest prediction probability
        highest_pred = str(np.argmax(outputs).tolist())
        pred_class = class_names[highest_pred]
    else: 
        pred_class = pred_class.obj #extract class name as text
        
    return(pred_class)

