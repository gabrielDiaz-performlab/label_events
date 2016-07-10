'''
gazeEventsDF has labelled data for all trials
When grabbing trialSourceDict, I must filter to grab data only from the selected trial
'''

from __future__ import division
import sys
sys.path.append("Modules/")
sys.path.append("../")
import os
import pandas as pd
import numpy as np
import bokeh.plotting as bkP
import bokeh.models as bkM

############################################################################
# Global parameters that influence the figure and labelling process

user = 'gdj'
gazeFileTime =  '2016-4-19-14-4'

trialNum = 10 # starts at 0
selectedBoxID = False  # keeps track of the selected box
nextBoxID = 0  # a unique identifier for new boxes


def create_glyphs():
    '''
    Create a dictionary of glyphs for rendering
    :return: a dictionary of glyphs
    '''

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

    saccadeGlyph = bkM.Rect(x='x',
                            y='y',
                            width='width',
                            height='height',
                            fill_alpha=0.6,
                            fill_color='green')

    fixationGlyph = bkM.Rect(x='x',
                             y='y',
                             width='width',
                             height='height',
                             fill_alpha=0.6,
                             fill_color='red')

    pursuitGlyph = bkM.Rect(x='x',
                            y='y',
                            width='width',
                            height='height',
                            fill_alpha=0.6,
                            fill_color='yellow')

    glyphDict = {'initRectGlyph': initRectGlyph, 'undefinedGlyph': undefinedGlyph,
                 'selectedGlyph': selectedGlyph,
                 'saccadeGlyph': saccadeGlyph, 'fixationGlyph': fixationGlyph, 'pursuitGlyph': pursuitGlyph}

    return glyphDict


def dataframe_from_pickle():

    '''
    Imports a data frame from a pickle file
    :return:
    '''

    # Try to import the pickle
    if not os.path.exists(filePath + gazeFileTime):
        os.makedirs(filePath + gazeFileTime)
    try:
        storedDataDF = pd.read_pickle(pickleLoc)
        print 'Pickle imported from ' + pickleLoc
        return storedDataDF
    except:

        print 'No existing data found.  Starting from scratch. '
        return False



def find_source_of_selectedID():
    '''
    Find the parent source of a rect with boxID == selectedID
    :return: a dictionary key if found, else returns False
    '''

    result = [sourceName for sourceName, source in trialSourceDict.iteritems() if
              selectedBoxID in source.data['boxID']]

    if (len(result) == 0):
        return False
    else:
        return result[0]


def pickle_trialDataSource():
    '''
    trialDataSource is converted to a dataframe: gazeEventsDF, and then pickled
    return:  void
    '''

    global gazeEventsDF

    trialEventsDF = pd.DataFrame()

    for key, source in trialSourceDict.items():

        if key != "index":
            eventDF = source.to_df()
            eventDF['eventType'] = key
            trialEventsDF = pd.concat([eventDF, trialEventsDF], axis=0)

    if gazeEventsDF is False:
        pd.to_pickle(trialEventsDF, pickleLoc)
    else:
        # Remove old records from current trial from gazeEventsDF
        gazeEventsDF = gazeEventsDF[gazeEventsDF['trialNum'] != trialNum]
        # Add new data
        gazeEventsDF = pd.concat([gazeEventsDF, trialEventsDF], axis=0)
        pd.to_pickle(gazeEventsDF, pickleLoc)

def init_trial_datasources():

    '''
    Fills the columnDataSources in trialSourceDict data extracted from an imported dataframe.
    Does not modify undefined columnDataSources.

    :return: Void
    '''

    global nextBoxID

    if( gazeEventsDF is False ):

        print 'gazeEventsDF is False'

        # No data saved, so we start with empty datasources

        undefinedSource = bkM.ColumnDataSource(
            data=dict(x=[], y=[], width=[], height=[], startTime=[], endTime=[], boxID=[], trialNum=[]))
        fixSource = bkM.ColumnDataSource(
            data=dict(x=[], y=[], width=[], height=[], startTime=[], endTime=[], boxID=[], trialNum=[]))
        sacSource = bkM.ColumnDataSource(
            data=dict(x=[], y=[], width=[], height=[], startTime=[], endTime=[], boxID=[], trialNum=[]))
        purSource = bkM.ColumnDataSource(
            data=dict(x=[], y=[], width=[], height=[], startTime=[], endTime=[], boxID=[], trialNum=[]))

        aDictOfDataSources = {'undefined': undefinedSource,
                           'fixation': fixSource,
                           'saccade': sacSource,
                           'pursuit': purSource}

        return aDictOfDataSources

    else:

        # Extract trial data from dataFrame
        print 'Extracting trial data sources from gazeEventsDF'

        gb = gazeEventsDF.groupby(['eventType', 'trialNum'])

        aDictOfDataSources = dict()

        for key in ['undefined', 'saccade', 'pursuit', 'fixation']:

            if key != 'undefined' and (key, trialNum) in gb.groups.keys():

                print 'Key is: ' + key + ' Trial: ' + str(trialNum)

                # Initialize dict entry with data from dataframe
                aDictOfDataSources[key] = bkM.ColumnDataSource(gb.get_group((key, trialNum)).drop('eventType', axis=1))

                # The process creates an unwanted "index" key in the sources
                # lets remove it.
                del aDictOfDataSources[key].data['index']

            else:
                # Initialize dict entry with empty data
                aDictOfDataSources[key] = bkM.ColumnDataSource(
                    data=dict(x=[], y=[], width=[], height=[], startTime=[], endTime=[], boxID=[], trialNum=[]))

        # Set boxID to a unique value (highest observed +1)
        for key, source in aDictOfDataSources.items():
            if len(source.data['boxID']) > 0:
                nextBoxID = np.max(np.hstack([nextBoxID, np.max(source.data['boxID'])])) + 1

        return aDictOfDataSources


# def timeSeries(frametime_fr=None, yDataList=None, yLabel=None, legendLabels=None, yLims=[0, 300],
#                        events_fr=None, trialsStarts_tr=None, plotHeight=500, plotWidth=1000):
#     '''
#     Creates a time-series plot of gaze data with Bokeh.
#     dataFrame = a dataframe with field ['frameTime'], ['eventFlag'], and ['trialNumber']
#     yLabel = A label for the Y axis.
#     yDataList = A list of vectors to be plotted on the Y axis as a line
#     legendLabels = A list of names for data plotted on the Y axis
#     yMax = Height of Y axidafdataFrames
#     markEvents= Show vertical lines with labels at events in dataFrame['eventFlag']
#     markTrials=Show vertical lines with labels at start of each trial
#     '''
#     from bokeh.palettes import Spectral6
#
#     if (isinstance(yDataList, list) is False):
#         raise TypeError('yDataList should be a list of lists.  Try [yData].')
#
#     if (legendLabels and isinstance(legendLabels, list) is False):
#         raise TypeError('legendLabels should be a list of lists.  Try [yLabelList].')
#
#     #### Setup figure
#
#     yRange = bkM.Range1d(yLims[0], yLims[1])
#
#     p = bkP.figure(plot_width=plotWidth, plot_height=plotHeight, tools="xpan,reset,save,xwheel_zoom,resize,tap",
#                    y_range=[0, 500],
#                    x_range=[np.min(frametime_fr), np.max(frametime_fr)],
#                    x_axis_label='time (s)', y_axis_label=yLabel)
#
#     p.ygrid.grid_line_dash = [6, 4]
#
#     # p.x_range = bkM.Range1d(dataFrame['frameTime'].values[0], dataFrame['frameTime'].values[0]+2)
#     p.x_range = bkM.Range1d(np.min(frametime_fr), np.min(frametime_fr) + 2)
#     p.y_range = yRange
#
#     ######################################################################################################
#     ## Plot gaze velocity(s)
#
#     for yIdx, yData in enumerate(yDataList):
#         source = bkP.ColumnDataSource(data=dict(x=frametime_fr, y=yData))
#
#         p.line(x='x', y='y', source=source, line_width=3, alpha=.7, name='velocity')
#
#         # if (legendLabels and len(legendLabels) >= yIdx):
#         #     p.line(x='x', y='y', source=source, line_width=3, alpha=.7, name='gazeVel', color=Spectral6[yIdx],
#         #            legend=legendLabels[yIdx])
#         #     # p.line(frametime_fr,yData,line_width=3, alpha=.7,color=Spectral6[yIdx],legend=legendLabels[yIdx])
#         # else:
#         #     p.line(x='x', y='y', source=source, line_width=3, alpha=.7, name='gazeVel', color=Spectral6[yIdx])
#
#     ######################################################################################################
#     ### Annotate events
#
#     showHighBox = False
#
#     # if( type(events_fr) is pd.Series ):
#     if (events_fr.any()):
#         showHighBox = True
#         X = frametime_fr[np.where(events_fr > 2)] + .01
#         Y = [yLims[1] * .9] * len(X)
#         text = [str(event) for event in events_fr[np.where(events_fr > 2)]]
#
#         source = bkP.ColumnDataSource(data=dict(x=X, y=Y, text=text))
#         p.text(source=source, x='x', y='y', text='text', text_font_size='8pt', text_font='futura', name='eventLabels')
#
#         ## Vertical lines at events
#         x = [[X, X] for X in frametime_fr[np.where(events_fr > 2)]]
#         y = [[yLims[0], yLims[1] * .9]] * len(x)
#
#         source = bkP.ColumnDataSource(data=dict(x=x, y=y))
#         p.multi_line(xs='x', ys='y', source=source, color='red', alpha=0.6, line_width=2, name='eventLines')
#
#         if (showHighBox):
#             high_box = bkM.BoxAnnotation(plot=p, bottom=yLims[1] * .9,
#                                          top=yLims[1], fill_alpha=0.7, fill_color='green', level='underlay')
#             p.renderers.extend([high_box])
#
#         return p


def make_trial_figure():  # trialNum, sessionDict, trialSourceDict, glyphDict):

    '''
    Create figure with velocity trace for the current trialNum
    :return: a bokeh figure
    '''

    # global sessionDict,,glyphDict

    gbTrial = eyeTrackingDataDict['raw'].groupby(['trialNumber'])
    gbProc  = eyeTrackingDataDict['processed'].groupby(['trialNumber'])

    frametime_fr = np.array(gbTrial['frameTime'].get_group(trialNum).diff().cumsum().values, dtype=np.float)
    frametime_fr[0] = 0

    fig = bkP.figure(plot_width=1000, plot_height=500,
                   y_range=[0, 400],
                   x_range=[np.min(frametime_fr), np.max(frametime_fr)],
                   x_axis_label='time (s)', y_axis_label='velocity')


    fig.ygrid.grid_line_dash = [6, 4]

    ######################################################################################################
    ## Plot gaze velocity(s)
    source = bkP.ColumnDataSource(data=dict(x=frametime_fr, y=gbProc.get_group(trialNum)[('cycSGVel', '2D')]))
    fig.line(x='x', y='y', source=source, line_width=3, alpha=.7, name='velocity')

    # fig = timeSeries(frametime_fr=frametime_fr, #gbTrial['frameTime'].get_group(trialNum).values,
    #                yDataList=[gbProc.get_group(trialNum)[('cycSGVel', '2D')]],
    #                events_fr=gbProc.get_group(trialNum)['eventFlag'].values,
    #                yLabel='gaze angular velocity',
    #                yLims=[0, 200],
    #                plotHeight=300)


    fig.tools = []

    undefinedSelectionGlyph = fig.add_glyph(trialSourceDict['undefined'],
                                            glyphDict['initRectGlyph'],
                                            selection_glyph=glyphDict['selectedGlyph'],
                                            nonselection_glyph=glyphDict['undefinedGlyph'],
                                            name='undefined')

    sacSelectionGlyph = fig.add_glyph(trialSourceDict['saccade'],
                                      glyphDict['saccadeGlyph'],
                                      selection_glyph=glyphDict['selectedGlyph'],
                                      nonselection_glyph=glyphDict['saccadeGlyph'],
                                      name='saccade')

    fixSelectionGlyph = fig.add_glyph(trialSourceDict['fixation'],
                                      glyphDict['fixationGlyph'],
                                      selection_glyph=glyphDict['selectedGlyph'],
                                      nonselection_glyph=glyphDict['fixationGlyph'],
                                      name='fixation')

    purSelectionGlyph = fig.add_glyph(trialSourceDict['pursuit'],
                                      glyphDict['pursuitGlyph'],
                                      selection_glyph=glyphDict['selectedGlyph'],
                                      nonselection_glyph=glyphDict['pursuitGlyph'],
                                      name='pursuit')

    # TODO: Remove falseRenderer.  It is a hack
    # I dont' want ot use the boxselect tool selection/hit testing.
    # Ideally, I woudl turn selection off.  It turns out that the best way to do this is to set
    # renderers explicitly.  It does not like an empty list, so I give it this single value.
    falseRenderer = fig.circle(x=1000, y=1000)

    fig.add_tools(bkM.BoxSelectTool(dimensions=["width"], renderers=[falseRenderer]))
    fig.add_tools(bkM.PanTool(dimensions=["width"]))
    fig.add_tools(bkM.WheelZoomTool(dimensions=["width"]))
    fig.add_tools(bkM.TapTool())

    fig.tool_events.on_change('geometries', box_callback)

    return fig


def box_callback(attr, old, new):
    '''
    A callback function for the box select tool
    '''

    global trialSourceDict, nextBoxID, selectedBoxID

    # Test to see if there is a geometry
    if (('y1' in new[0].keys()) == False):
        # print 'No geometry'
        return

    width = new[0]['x1'] - new[0]['x0']
    height = new[0]['y1'] - new[0]['y0']
    x = new[0]['x0'] + width / 2
    y = new[0]['y0'] + height / 2

    startTime = new[0]['x0']
    endTime = new[0]['x1']

    new_data = dict(trialSourceDict['undefined'].data)

    new_data['x'].append(x)
    new_data['y'].append(y)
    new_data['width'].append(width)
    new_data['height'].append(height)
    new_data['startTime'].append(startTime)
    new_data['endTime'].append(endTime)
    new_data['boxID'].append(nextBoxID)
    new_data['trialNum'].append(trialNum)


    # update entire .data in one go
    trialSourceDict['undefined'].data = new_data

    # explicit trigger
    trialSourceDict['undefined'].trigger('data', new_data, new_data)

    selectedBoxID = nextBoxID
    nextBoxID += 1


def tap_callback(source, valueIdx):
    '''
    Note that the tap callback is executed once per data source attached to the figure.
    So, we have some checks to make sure that ...
    '''

    global selectedBoxID

    if( valueIdx ):
        print 'tap_callback: selectedBoxID = ' + str(selectedBoxID)
        selectedBoxID = source.data['boxID'][valueIdx]
    #except:
    #    print 'tap_callback: In tap_callback(), valueidx = ' + str(valueIdx)


def selection_cb(sourceKey,selected):

    if (len(selected['1d']['indices']) == 0):
        # Hit test, but nothing actually selected
        return
    else:
        tap_callback(trialSourceDict[sourceKey], selected['1d']['indices'][0])


def undefined_selected(attr, old, new):
    selection_cb('undefined', new)


def fixation_selected(attr, old, new):
    selection_cb('fixation', new)


def saccade_selected(attr, old, new):
    selection_cb('saccade', new)


def pursuit_selected(attr, old, new):
    selection_cb('pursuit',new)

def find_source_of_selectedID():
    '''
    Finds the parent source of selectedBoxID
    :return: a dictionary key: 'undefined', 'saccade', 'fixation', or, 'pursuit'
    '''

    def find_in_list_of_list(mylist, char):
        for sub_list in mylist:
            if char in sub_list:
                return mylist.index(sub_list)
        return False

    boxIDs_source = [source.data['boxID'] for source in trialSourceDict.values()]

    sourceIdx = find_in_list_of_list(boxIDs_source, selectedBoxID)
    sourceKey = trialSourceDict.keys()[sourceIdx]

    if sourceIdx is not False:
        return sourceKey
    else:
        return False, False

def move_between_sources(oldSourceKey, newSourceKey):
    '''

    :param oldSource: The source in which the rect with self.selectedID lies
    :param newSource: The source in which the rect with self.selectedID lies
    :return: void
    '''

    global trialSourceDict

    oldSource = trialSourceDict[oldSourceKey]
    newSource = trialSourceDict[newSourceKey]

    # Get the index of the selected item in currSource
    valueIdx = oldSource.data['boxID'].index(selectedBoxID)

    oldData = oldSource.data
    newData = newSource.data

    for key, values in oldData.iteritems():
        # print currSource.data[key][valueIdx]
        newData[key].append(oldData[key][valueIdx])
        del values[valueIdx]

    trialSourceDict[oldSourceKey].data = oldData
    trialSourceDict[oldSourceKey].trigger('data', oldData, oldData)

    trialSourceDict[newSourceKey].data = newData
    trialSourceDict[newSourceKey].trigger('data', newData, newData)

    # According to the folks at continuum, the .trigger method should not be necessary.
    # However, without it, nothing changes in the figure.

    pickle_trialDataSource()

def apply_label(newSourceLabel):
    # This assumes that data for the selected event belongs to undefinedSource.
    # This may not be the case.

    global selectedBoxID
    global trialSourceDict

    if selectedBoxID is False:
        print 'Nothing selected, or ID of selected not valid: ' + str(selectedBoxID)
        return

    print 'apply_label(): selectedBoxID: ' + str(selectedBoxID)
    oldSourceLabel = find_source_of_selectedID()

    move_between_sources(oldSourceLabel, newSourceLabel)

    selectedBoxID = False


def remove_cb():

    global trialSourceDict, selectedBoxID

    sourceKey = find_source_of_selectedID()
    listIdx_of_selected = trialSourceDict[sourceKey].data['boxID'].index(selectedBoxID)

    selectedBoxID = False

    newData = trialSourceDict[sourceKey].data

    #  For each key in the source, remove the values of entry valueIdx
    for key, values in newData.iteritems():
        del values[listIdx_of_selected]

    trialSourceDict[sourceKey].data = newData
    trialSourceDict[sourceKey].trigger('data', newData, newData)


def label_saccade_cb():
    apply_label('saccade')

def label_pursuit_cb():
    apply_label('pursuit')

def label_fixation_cb():
    apply_label('fixation')

def trial_text_cb(attr, old, new):

    global trial_text

    try:
        int(new)
    except:
        print 'trial text value is invalid'
        trial_text.value = old
        return

    global trialNum

    if int(new) > numTrials :
        trial_text.value = str(numTrials)-1
        trialNum = numTrials-1

    elif int(new) < 0:
        trial_text.value = str(0)
        trialNum = 0

    else:
        trial_text.value = new
        trialNum = int(new)

    changeTrial()

def force_update_sources():
    global trialSourceDict
    [source.trigger('data', source.data, source.data) for source in trialSourceDict.values()]

def print_sources():

    def print_source(dataSourceTuple):
        print dataSourceTuple[0] + ': ' + str(dataSourceTuple[1].data)

    [print_source(source) for source in trialSourceDict.items()]

def changeTrial():

    print 'changeTrial: New trial # is ' + str(trialNum)

    global fig, trialSourceDict

    gbTrial = eyeTrackingDataDict['raw'].groupby(['trialNumber'])
    gbProc = eyeTrackingDataDict['processed'].groupby(['trialNumber'])

    frametime_fr = np.array(gbTrial['frameTime'].get_group(trialNum).diff().cumsum().values, dtype=np.float)
    frametime_fr[0] = 0

    velocity_fr = gbProc.get_group(trialNum)[('cycSGVel', '2D')]
    events_fr = gbTrial.get_group(trialNum)['eventFlag'].values

    # Update velocity
    target = 'velocity'
    new_data = fig.select(target)[0].data_source.data
    new_data['x'] = frametime_fr
    new_data['y'] = velocity_fr
    fig.select(target)[0].data_source.data = new_data

    #fig.x_range = bkM.Range1d(frametime_fr[0], frametime_fr[-1])


    print '***** BEFORE *****\n'
    print_sources()

    trialSourceDict = init_trial_datasources()
    #[source.trigger('data', source.data, source.data) for source in trialSourceDict.values()]

    #new_data = trialSourceDict['saccade'].data
    #trialSourceDict['saccade'].trigger('data', new_data, new_data)

    force_update_sources()
    print '***** AFTER *****\n'
    print_sources()

    # Setup callbacks for tools and sources
    trialSourceDict['undefined'].on_change('selected', undefined_selected)
    trialSourceDict['saccade'].on_change('selected', saccade_selected)
    trialSourceDict['pursuit'].on_change('selected', pursuit_selected)
    trialSourceDict['fixation'].on_change('selected', fixation_selected)

    global selectedBoxID
    selectedBoxID = False

    # fig = make_trial_figure()
    #
    # # Add figure and widgets to a layout
    # ###########################################################################
    # from bokeh.layouts import layout
    # layout = layout([[fig], [widgetBox]], sizing_mode='fixed')
    #
    # ###########################################################################
    # # Add layout to current document
    # from bokeh.io import curdoc, set_curdoc
    #
    # curdoc().clear()
    # curdoc().add_root(layout)

    # target = 'eventLabels'
    # new_data = fig.select(target)[0].data_source.data
    # new_data['x'] = frametime_fr[np.where(events_fr > 2)] + .01
    # new_data['y'] = [fig.y_range.end * .9] * len(frametime_fr[np.where(events_fr > 2)])
    # fig.select(target)[0].data_source.data = new_data
    #
    # target = 'eventLines'
    # new_data = fig.select(target)[0].data_source.data
    # new_data['x'] = [[x, x] for x in frametime_fr[np.where(events_fr > 2)]]
    # new_data['y'] = [[fig.y_range.start, fig.y_range.end * .9]] * sum(events_fr > 2)
    # fig.select(target)[0].data_source.data = new_data


# def add_widgets():
#
#     from bokeh.models.widgets import Button, TextInput
#
#     label_saccade_button = Button(label='saccade')
#     label_saccade_button.on_click(label_saccade_cb)
#
#     label_pursuit_button = Button(label='pursuit')
#     label_pursuit_button.on_click(label_pursuit_cb)
#
#     label_fixation_button = Button(label='fixation')
#     label_fixation_button.on_click(label_fixation_cb)
#
#     remove_button = Button(label='remove')
#     remove_button.on_click(remove_cb)
#
#     trial_text = TextInput(value=str(trialNum), title="Trial: ")
#     trial_text.on_change('value', trial_text_cb)
#
#     widgets = [label_saccade_button, label_pursuit_button, label_fixation_button, remove_button, trial_text]
#
#     return widgets

############################################################################

#  Directories, etc

filePath = 'LabelledGazeEvents/'
pickleName = gazeFileTime + '_' + user + '.pickle'
pickleLoc = filePath + gazeFileTime + '/' + pickleName

###########################################################################
# Load datafile with velocity timeseries

eyeTrackingDataDict = pd.read_pickle('Data/' + gazeFileTime + '/' + 'exp_data-' + gazeFileTime + '-proc.pickle')
numTrials = len(np.unique(eyeTrackingDataDict['processed']['trialNumber']))

###########################################################################
# Get stored labels / rectangles
gazeEventsDF = dataframe_from_pickle()

###########################################################################
# Create figure using velocity timeseries and labelled events (if any)
trialSourceDict = init_trial_datasources()
print_sources()

glyphDict = create_glyphs()
fig = make_trial_figure()

# Setup callbacks for tools and sources
trialSourceDict['undefined'].on_change('selected', undefined_selected)
trialSourceDict['saccade'].on_change('selected', saccade_selected)
trialSourceDict['pursuit'].on_change('selected', pursuit_selected)
trialSourceDict['fixation'].on_change('selected', fixation_selected)

###########################################################################
# Add widgets and their callbacks
#widgets = add_widgets()

from bokeh.models.widgets import Button, TextInput

label_saccade_button = Button(label='saccade')
label_saccade_button.on_click(label_saccade_cb)

label_pursuit_button = Button(label='pursuit')
label_pursuit_button.on_click(label_pursuit_cb)

label_fixation_button = Button(label='fixation')
label_fixation_button.on_click(label_fixation_cb)

remove_button = Button(label='remove')
remove_button.on_click(remove_cb)

trial_text = TextInput(value=str(trialNum), title="Trial: ")
trial_text.on_change('value', trial_text_cb)

widgets = [label_saccade_button, label_pursuit_button, label_fixation_button, remove_button, trial_text]

from bokeh.layouts import widgetbox
widgetBox = widgetbox(*widgets, sizing_mode='fixed')

# Add figure and widgets to a layout
###########################################################################
from bokeh.layouts import layout
layout = layout( [[fig],[widgetBox]] , sizing_mode='fixed')

###########################################################################
# Add layout to current document
from bokeh.io import curdoc
curdoc().add_root(layout)

from bokeh.client import push_session
session = push_session(curdoc())
session.show()  # open the document in a browser

session.loop_until_closed()  # run forever