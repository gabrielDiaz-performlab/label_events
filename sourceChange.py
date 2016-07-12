from __future__ import division
import sys
sys.path.append("Modules/")
sys.path.append("../")
import pandas as pd
import numpy as np
import bokeh.plotting as bkP
import bokeh.models as bkM


def create_glyphs():
    '''
    Create a dictionary of glyphs for rendering
    :return: a dictionary of glyphs
    '''

    # noinspection PyPep8Naming
    initRectGlyph = bkM.Rect(x='x',
                             y='y',
                             width='width',
                             height='height',
                             fill_alpha=0.6,
                             fill_color='gray')

    undefinedGlyph = bkM.Rect(x='x',
                              y='y',
                              width='width',
                              height='height',
                              fill_alpha=0.6,
                              fill_color='gray')

    selectedGlyph = bkM.Rect(x='x',
                             y='y',
                             width='width',
                             height='height',
                             fill_alpha=0.6,
                             fill_color='cyan')

    glyphDict = {'initRectGlyph': initRectGlyph, 'undefinedGlyph': undefinedGlyph,
                 'selectedGlyph': selectedGlyph}

    return glyphDict

def create_fig():  # trialNum, sessionDict, trialSourceDict, glyphDict):

    '''
    Create figure with velocity trace for the current trialNum
    :return: a bokeh figure
    '''

    # global sessionDict,,glyphDict

    fig = bkP.figure(plot_width=1000, plot_height=500,
                     y_range=[-1, 1],
                     x_axis_label='time (s)', y_axis_label='velocity')

    fig.ygrid.grid_line_dash = [6, 4]

    ######################################################################################################
    ## Plot gaze velocity(s)

    x = np.deg2rad(range(720))
    y = np.sin(x)

    source = bkP.ColumnDataSource(data=dict(x=x, y=y))
    fig.line(x='x', y='y', source=source, line_width=3, alpha=.7, name='velocity')


    fig.tools = []

    global glyphDict, undefinedSource

    undefinedSelectionGlyph = fig.add_glyph(undefinedSource,
                                            glyphDict['initRectGlyph'],
                                            selection_glyph=glyphDict['selectedGlyph'],
                                            nonselection_glyph=glyphDict['undefinedGlyph'],
                                            name='undefined')

    fig.add_tools(bkM.PanTool(dimensions=["width"]))
    fig.add_tools(bkM.WheelZoomTool(dimensions=["width"]))

    return fig

def update_figure_A():

	global fig, undefinedSource

	print 'Before: ' + str(undefinedSource.data)

	new_data = dict(x=[2, 6, 10], y=[0, 0, 0], width=[0.5, 0.5, 0.5], height=[2, 2, 2])

	undefinedSource.data = new_data
	undefinedSource.trigger('data', new_data, new_data)

	print 'After: ' + str(undefinedSource.data) + '\n'

def update_figure_B():

	global fig, undefinedSource

	print 'Before: ' + str(undefinedSource.data)

	new_data = data=dict(x=[], y=[], width=[], height=[])
	undefinedSource.data = new_data

	undefinedSource.trigger('data', new_data, new_data)

	print 'After: ' + str(undefinedSource.data) + '\n'

###########################################################################
# Setup figure, glyphs, and a datasource

glyphDict = create_glyphs()
undefinedSource = bkM.ColumnDataSource(data=dict(x=[0,4,8], y=[0,0,0], width=[0.25,0.25,0.25], height=[2,2,2]))
fig = create_fig()

###########################################################################
# Add widgets and their callbacks

from bokeh.models.widgets import Button, TextInput

changeA_button = Button(label='UpdateFig A')
changeA_button.on_click(update_figure_A)

changeB_button = Button(label='UpdateFig B')
changeB_button.on_click(update_figure_B)

widgets = [changeA_button, changeB_button]

from bokeh.layouts import widgetbox
widgetBox = widgetbox(*widgets, sizing_mode='fixed')

# Add figure and widgets to a layout
###########################################################################
from bokeh.layouts import layout
layout = layout( [[fig], [widgetBox]], sizing_mode='fixed')

###########################################################################
# Add layout to current document

from bokeh.io import curdoc
curdoc().add_root(layout)

from bokeh.client import push_session
session = push_session(curdoc())
session.show()  # open the document in a browser

session.loop_until_closed()  # run forever