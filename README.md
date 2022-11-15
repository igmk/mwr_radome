# Microwave Radiometer "wet" flagging and radome monitoring
This is a toolbox to perform quality checks on 
ground-based microwave radiometer (MWR) observations.
Generally speaking this code provides an additional 
flagging of biased MWR brightness temperature (TB)
data during and after rain events. 

The additional flag and a status flag for the radome 
hygroscopic properties is added to the data. The radome 
status flag can help find the optimal time to replace
the radome.

This toolset should work with the HDCP2 SAMD data format 
and the E-PROFILE data format. It is also functional
with the standard RPG (HATPRO) data format, however this is
not recommended. The RPG spectral retrieval (SPC) is currently
required. 

## Prerequisites
The code has been tested with python 3.6 and above and
on Unix based machines. However, you will find it easy
to integrate into your pipeline.

The following packages are required:

```
xarray, collections, datetime, glob, typing, shutil, re
```
## Getting started

Take a look into the example in ./envVars/rao_mwr02/. 
For each instrument you would like to check, please define
all the variables listed in the examples.
The output will then be saved in a similar folder structure
with "orig" replaced by "qc_done" or "radome".

Put your environment-variable-json-files in a folder
with the same name as your station (which you can choose
freely.). 

The variable "timePeriodUpdate" can be set to "lastXDays",
with X replaced by an integer. OR by a time period similar to
["2022-01-31", "2022-12-31"]. In this way you can process a larger 
dataset or run a cron job to give you a regular update.

In order to run call

```
python3 -m MWR_radome <station>
```

in case of the given example
```
python3 -m MWR_radome rao_mwr02
```



