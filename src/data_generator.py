from math import sqrt, sin, cos
import pandas as pd

DATASET_SIZE = 10000

def load_dataset():
    dataset = pd.read_csv('dataset.csv')
    return dataset

def save_dataset():
    df = prepare_dataset(DATASET_SIZE)
    df.to_csv('dataset.csv', index = False)

def prepare_dataset(set_size):
    points = generate_points(set_size)
    #create dataframe from generated points
    data = pd.DataFrame.from_records(points)
    data.columns = ['x', 'y', 'z']
    return data

def generate_points(points_number):
    # f(x,y) = sin(x)*cos(5*y)/5

    # generate range of area to generate points, x and y both in range [-sphere_range, sphere_range)
    sphere_range = int(sqrt(points_number)/2)
    xmin = ymin = -sphere_range
    xmax = ymax = sphere_range

    points = []
    for x in range(xmin, xmax, 1):
        for y in range(ymin, ymax, 1):
            z = sin(x)*cos(5*y)/5 # calculate z coordinate with given f(x,y)
            points.append([x,y,z]) 

    return points
