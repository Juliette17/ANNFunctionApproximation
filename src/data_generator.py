from math import sqrt, sin, cos
import pandas as pd

DATASET_SIZE = 10000
DENSITY = 0.1

def load_dataset():
    dataset = pd.read_csv('dataset2.csv')
    return dataset

def save_dataset():
    df = prepare_dataset(DATASET_SIZE)
    df.to_csv('dataset2.csv', index = False)

def prepare_dataset(set_size):
    points = generate_points(set_size)
    #create dataframe from generated points
    data = pd.DataFrame.from_records(points)
    data.columns = ['x', 'y', 'z']
    return data

def generate_points(points_number=DATASET_SIZE, density=DENSITY):
    # f(x,y) = 5*sin(0.1*x)*cos(0.5*y)

    # generate range of area to generate points, x and y both in range [-sphere_range, sphere_range)
    sphere_range = int(sqrt(points_number)/2)
    xmin = ymin = -sphere_range
    xmax = ymax = sphere_range

    points = []
    for x in range(xmin, xmax, 1):
        for y in range(ymin, ymax, 1):
            z = 5*sin(0.1*x*density)*cos(0.5*y*density) # calculate z coordinate with given f(x,y)
            points.append([x*density,y*density,z]) 

    return points

# uncomment to generate dataset
save_dataset()