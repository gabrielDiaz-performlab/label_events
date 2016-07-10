# label_events
A bokeh app to enable the manual labelling of events in time series data.

Currently, this runs in app mode.  
To run, open terminal>  bokeh serve --show eventlabeller.py

To switch to client mode, uncomment the bottom-most lines of code:

from bokeh.client import push_session
session = push_session(curdoc())
session.show()  # open the document in a browser
session.loop_until_closed()  # run forever
