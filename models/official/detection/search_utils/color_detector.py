##################
# input: filename, binary mask
# output: list of dominant colors in the masked area
##################
##################
# Color Detector algorithm that uses an image and a binary mask over it to detect the two most dominant colors present in the masked area.
# The dominant colors are detected using the CSS4 color catalog and simplified to more common names
# The algorithm makes a color histogram out of each pixel value with a bucket size 20 and finds the nearest color in the colro catalog
##################

import numpy as np
from PIL import Image

import webcolors
import matplotlib.colors as mc

common_colors = {}
common_colors['red'] = ['coral', 'crimson', 'darkred', 'darksalmon', 'firebrick', 'indianred', 'lightcoral', 'lightsalmon', 'maroon', 'mistyrose', 'red', 'salmon', 'sienna', 'tomato']
common_colors['blue'] = ['aliceblue', 'aqua', 'aquamarine', 'azure', 'blue', 'blueviolet', 'cadetblue', 'cornflowerblue', 'cyan', 'darkblue', 'darkcyan', 'darkslateblue', 'darkturquoise', 'deepskyblue', 'dodgerblue', 'lightblue', 'lightcyan', 'lightskyblue', 'lightsteelblue', 'mediumaquamarine', 'mediumblue', 'mediumslateblue', 'mediumturquoise', 'midnightblue', 'navy', 'paleturquoise', 'powderblue', 'royalblue', 'skyblue', 'slateblue', 'steelblue', 'teal', 'turquoise']
common_colors['yellow'] = ['cornsilk', 'darkgoldenrod', 'gold', 'goldenrod', 'lemonchiffon', 'lightgoldenrodyellow', 'lightyellow', 'palegoldenrod', 'yellow']
common_colors['green'] = ['chartreuse', 'darkgreen', 'darkkhaki', 'darkolivegreen', 'darkseagreen', 'forestgreen', 'green', 'greenyellow', 'honeydew', 'khaki', 'lawngreen', 'lightgreen', 'lightseagreen', 'lime', 'limegreen', 'mediumseagreen', 'mediumspringgreen', 'mintcream', 'olive', 'olivedrab', 'palegreen', 'seagreen', 'springgreen', 'yellowgreen']
common_colors['black'] = ['black']
common_colors['white'] = ['antiquewhite', 'floralwhite', 'ghostwhite', 'ivory', 'navajowhite', 'snow', 'white', 'whitesmoke']
common_colors['grey'] = ['darkgray', 'darkgrey', 'darkslategray', 'darkslategrey', 'dimgray', 'dimgrey', 'gainsboro', 'gray', 'grey', 'lightgray', 'lightgrey', 'lightslategray', 'lightslategrey', 'slategray', 'slategrey']
common_colors['brown'] = ['brown', 'burlywood', 'chocolate', 'peru', 'rosybrown', 'saddlebrown', 'sandybrown']
common_colors['cream'] = ['beige', 'bisque', 'blanchedalmond', 'linen', 'oldlace', 'moccasin', 'papayawhip', 'peachpuff', 'seashell', 'tan', 'wheat']
common_colors['purple'] = ['darkmagenta', 'darkorchid', 'darkviolet', 'fuchsia', 'indigo', 'lavender', 'lavenderblush', 'magenta', 'mediumorchid', 'mediumpurple', 'mediumvioletred', 'orchid', 'palevioletred', 'plum', 'purple', 'rebeccapurple', 'thistle', 'violet'] 
common_colors['orange'] = ['darkorange', 'orange', 'orangered']
common_colors['pink'] = ['deeppink', 'hotpink', 'lightpink', 'pink']
common_colors['silver'] = ['silver']

def closest_colour(requested_colour):
    min_colours = {}
    for name, key in mc.CSS4_COLORS.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_dominant_color(filename, mask):
    img = Image.open(filename)
    ni = np.array(img)
    counter = {}
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                r, g, b = ni[i][j]
                r = r - (r%20)
                g = g - (g%20)
                b = b - (b%20)
                c = (r,g,b)
                if c not in counter:
                    counter[c] = 0
                counter[c] += 1
    colors = [[c,counter[c]] for c in counter]
    colors.sort(key=lambda x:x[1], reverse=True)
    d_c = list(set([closest_colour(c[0]) for c in colors[:2]]))
    dominant_colors = []
    for c in d_c:
        check = True
        for common_color in common_colors:
            if c in common_colors[common_color]:
                dominant_colors.append(common_color)
                check = False
        if check:
            dominant_colors.append(c)
    return list(set(dominant_colors))