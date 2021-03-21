import numpy as np
import matplotlib.pyplot as plt
import math

color_list_19 = [(128, 64, 128), (244, 36, 232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), 
    (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), 
    (  0,  0,142), (  0,  0, 70), (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)]

def apply_color_to_map (semantic_map, num_classes=10):
    assert len(semantic_map.shape) == 2
    H, W = semantic_map.shape
    color_semantic_map = np.zeros((H, W, 3), dtype='uint8')
    for i in range(num_classes):
        color_semantic_map[semantic_map==i] = color_list_19[i]
    return color_semantic_map

ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}
dict_roomType2id = dict(zip(list(ROOM_CLASS.values()), range(len(ROOM_CLASS.values()))))

def visualize_house(house):
    room_types = house[0]
    rooms = house[1]
    edge_to_room = house[3]

    # find top left and right bottom corner of the house
    min_x0, min_y0, max_x1, max_y1 = math.inf, math.inf, -math.inf, -math.inf
    for room in rooms:
        x0, y0, x1, y1 = room
        if min_x0 > x0:
            min_x0 = x0
        if min_y0 > y0:
            min_y0 = y0
        if max_x1 < x1:
            max_x1 = x1
        if max_y1 < y1:
            max_y1 = y1
    min_x0, min_y0, max_x1, max_y1 = min_x0-5, min_y0-5, max_x1+5, max_y1+5

    floormap = np.ones((max_y1-min_y0, max_x1-min_x0), dtype=int)*255
    for i, room in enumerate(rooms):
        x0, y0, x1, y1 = room
        x0 = x0 - min_x0
        y0 = y0 - min_y0
        x1 = x1 - min_x0
        y1 = y1 - min_y0
        type_id = dict_roomType2id[int(room_types[i])]
        #print('type_id = {}'.format(type_id))
        floormap[y0+5:y1+5, x0+5:x1+5] = type_id
        #assert 1==2

    color_floormap = apply_color_to_map(floormap)
    return color_floormap