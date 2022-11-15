# -*- coding: utf-8 -*-
# --------------------------------
# Name:       functions.py
# Purpose:    Collection of useful functions.
# Author:     Moritz Loeffler
# Created:    2022-11-11
# Python Version:   3.6
# Version:    1
# Last Edit:  2022-11-14
# --------------------------------
"""
Collection of functions that are repeated across all modules.
"""
import sys
import json
import os
import re

import numpy as np
import datetime as dt
import shutil
import xarray as xr
from typing import Union, List, Any

envVarsDir = os.path.dirname(__file__) + "/envVars"


def getEnvVars(directory: str = envVarsDir, fname: str = "/env_vars_", names: Union[List[str], str] = '') -> dict:
    """
    Retrieve env vars from fname = ./env_vars.json
    """
    # fname = "/env_vars.json"
    envvars = dict()
    if type(names) == list:
        for name in names:
            envvars = dict(envvars, **getEnvVars(directory, fname=fname,
                                                 names=name))
    elif type(names) == str:
        path = directory + fname + names + '.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                try:
                    envvars = dict(**json.load(f))
                except json.decoder.JSONDecodeError:
                    print(path)
                    raise
        else:
            sys.exit("Environment variable json file not found: %s" % path)
    return envvars


def copy(src: str, dst: str):
    """Copy files without permission settings."""
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    shutil.copyfile(src, dst)


def findNearest(array: Union[list, np.array], value: Any) -> Any:
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


def createDir(directory: str):
    """Create directory of it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def getTimePeriod(timePeriod: Union[str, List[str]]) -> List[np.datetime64]:
    """Return a list with beginning and end of evaluated time period."""
    if type(timePeriod) == str:
        if "Days" in timePeriod:
            try:
                N = int(re.search(r'\d+', timePeriod).group())
            except AttributeError:
                N = 2
            now = np.datetime64(dt.datetime.utcnow()).astype('datetime64[D]')
            begin = (now - np.timedelta64(N, 'D')).astype('datetime64[D]')
            timePeriod = [begin, now - np.timedelta64(1, 'm')]
        elif timePeriod == 'currentMonth':
            now = np.datetime64(dt.datetime.utcnow()).astype('datetime64[h]')
            begin = (np.datetime64(dt.datetime.utcnow()) - np.timedelta64(1, 'D')).astype('datetime64[M]')
            timePeriod = [begin, now, begin + np.timedelta64(1, 'M') - np.timedelta64(1, 'm')]
    else:
        timePeriod = [np.datetime64(timePeriod[0]), np.datetime64(timePeriod[1])]
    return timePeriod


def createBinaryArray(array: np.array, maxBit: int) -> np.ndarray:
    """Return 2D array with bits as 0 and 1 accordingly."""
    noOfTimes = len(array)
    noOfBits = len(np.binary_repr(int(maxBit)))
    possibleBits = np.arange(noOfBits) + 1
    # creating binary grid
    binaryArray = np.zeros((noOfTimes, noOfBits))
    for i in np.arange(len(array)):
        f = array[i]
        try:
            bits = list(np.binary_repr(int(f)))
        except ValueError:
            continue
        for j in possibleBits:
            try:
                binaryArray[i, -j] = float(bits[-j]) * j
            except IndexError:
                pass
    return binaryArray


def dropAdditionalTimeDims(ds: xr.Dataset, dsLast: xr.Dataset) -> xr.Dataset:
    """Drop time dim from variables, where it was added by concat."""
    try:
        dsTemp = ds.copy()
        ds = ds.drop_dims('time')
        keys = list(dsTemp.keys())
        for key in keys:
            if 'time' in list(dsLast[key].dims):
                ds[key] = dsTemp[key]
            else:
                ds[key] = dsLast[key]
    except AttributeError as e:
        if "NoneType" in str(e):
            pass
        else:
            raise
    return ds