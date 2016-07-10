# label_events
A bokeh app to enable the manual labelling of events in time series data.

Currently, this runs in app mode.  
To run, open terminal>  bokeh serve --show eventlabeller.py

To switch to client mode, uncomment the last 4 lines of code in the file, underneate from bokeh.client import push_session.
