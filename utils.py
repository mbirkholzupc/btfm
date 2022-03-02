import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

class GID:
    """
    Global ID class. Used to generate a unique ID each time it is called.
    Note that this function is not thread-safe.
    """
    def __init__(self, val=0):
        self._id = val
    def next(self):
        tmp = self._id
        self._id += 1
        return tmp
    def rollback(self):
        self._id -= 1

def plotOnImage(img, data, fmt):
    # Need to draw image into array instead of plot it
    # See agg_buffer_to_array reference page in matplotlib documentation
    # Also, had to jump through a lot of hoops to get rid of the white space around the image
    arr_fig = Figure(figsize=plt.figaspect(img))
    canvas = FigureCanvas(arr_fig)
    arr_fig.subplots_adjust(0,0,1,1)
    arr_ax = arr_fig.subplots()
    arr_ax.set_xticks([])
    arr_ax.set_yticks([])
    arr_ax.imshow(img)
    arr_ax.plot(data[:,1],data[:,0],fmt)
    # Draw image to buffer
    canvas.draw()
    out_img = np.array(canvas.renderer.buffer_rgba())
    return out_img

def filterJoints(joints, visibility):
    fjoints=joints[visibility>0,:]
    return fjoints

def plotMultiOnImage(img, data_fmt_zip, rects=None):
    # Need to draw image into array instead of plot it
    # Same as plotOnImage except data and fmt are a zipped list of points and formats
    arr_fig = Figure(figsize=plt.figaspect(img))
    canvas = FigureCanvas(arr_fig)
    arr_fig.subplots_adjust(0,0,1,1)
    arr_ax = arr_fig.subplots()
    arr_ax.set_xticks([])
    arr_ax.set_yticks([])
    arr_ax.imshow(img)
    for data, fmt in data_fmt_zip:
        arr_ax.plot(data[:,1],data[:,0],fmt)

    if rects:
        for r in rects:
            arr_ax.add_patch(r)

    # Draw image to buffer
    canvas.draw()
    out_img = np.array(canvas.renderer.buffer_rgba())
    return out_img

def clip_detect(vals, minval, maxval):
    cliplow = vals < minval
    cliphigh = vals > maxval
    return (any(cliplow) or any(cliphigh))
