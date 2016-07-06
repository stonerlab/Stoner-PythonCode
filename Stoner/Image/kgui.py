# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:56:32 2016

@author: phyrct
"""

class KerrGUI():
    def _rect_selected(extents):
        """event function for skimage.viewer.canvastools.RectangleTool
        """
        rect_coord.update({'done': True})
         
    def draw_rectangle(self):
        """Draw a rectangle on the image and return the coordinates
        
        Returns
        -------
        box: ndarray
            [xmin,xmax,ymin,ymax]"""
            
        viewer=ImageViewer(self)
        viewer.show()
        from skimage.viewer.canvastools import RectangleTool
        rect_selected_yet={'done':False} #mutable object to store func status in
        rect_tool = RectangleTool(viewer, on_enter=_rect_selected)
        while not rect_selected_yet['done']:
            time.sleep(2)
            pass
        coords=np.int64(rect_tool.extents)
        viewer.close()
        return coords 
    
    def draw_trace(vert_coord, width=1):
        """Line trace horizontal at vertical coord averaging over width"""
        pass
    
    def plt_histogram(self, **kwarg):
        """plot histogram of image intensities, pass through kwarg to matplotlib.pyplot.hist"""
        plt.hist(self.ravel(), **kwarg)