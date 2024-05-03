# -*- coding: utf-8 -*-
# --------------------------------
# Name:       MWR_radome.py
# Purpose:    MWR flagging wet radome and monitoring radome condition.
# Author:     Moritz Loeffler
# Created:    2022-11-10
# Python Version:   3.6
# Version:    1.1
# Last Edit:  2024-05-03
# --------------------------------
"""
Data quality checks and monitoring of MWR devices with high resolution data.
The script accept data formats according to SAMD-Standard (HDCP2) and e-profile.
When providing observations in the RPG-format, no additional quality flag is added.
The spectral retrieval is expected in RPG Format.

"""

#########################
# Modules

import numpy as np
import xarray as xr
import sys
import os
from collections import OrderedDict
import datetime as dt
import glob
from typing import List, Optional, Tuple, Union
import shutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
try:
    from bottleneck import nanmean, nansum, nanmax, nanmedian
except ModuleNotFoundError:
    from numpy import nanmean, nansum, nanmax, nanmedian

sys.path.append(os.path.dirname(__file__))
import functions

# Concise Date formatter
converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[dt.date] = converter
munits.registry[dt.datetime] = converter

###  global variables

envVarsDir = os.path.dirname(__file__) + "/envVars"

class ProcessHandler(object):
    """Handle parallel processes and manage directories.
    """

    def __init__(self, stations: Union[str, List[str]], pid):
        """Initialize a new process and assign a process id.
        """
        self.stations = [stations]
        self.pid = pid
        self.envVarsDir = envVarsDir + '/' + str(pid)

    def manageEnvVars(self, ):
        """Copy envVars to process dir and store path in self.envVarsDir."""
        self.envVarsDir = envVarsDir + '/%03d' % self.pid
        envVarsFiles = glob.glob(envVarsDir + '/env_vars_*.json')
        functions.createDir(self.envVarsDir)
        for file in envVarsFiles:
            shutil.copy(file, self.envVarsDir + '/')


class MWRQuality(object):
    """Check the data quality of MWR data for time period."""

    def __init__(self, ph: ProcessHandler):
        """Run pipe"""
        self.ph = ph
        self.configureProcess()
        self.runQC()

    def configureProcess(self, ):
        """Load config files, and set up variables"""
        # Load envVars
        envVarsNames = ['general']
        self.__dict__ = dict(self.__dict__,
                             **functions.getEnvVars(directory=self.ph.envVarsDir,
                                                    names=envVarsNames))
        self.timePeriod = functions.getTimePeriod(self.timePeriodUpdate)
        # Set paths
        self.radomeDayFName = "/wet_radome_day_00.nc"
        self.radomeMonthFname = "{:/wet_radome_%y%m.nc}"
        self.currentRadome = np.datetime64('1970-01-01')
        self.station = self.ph.stations[0]

    def runQC(self):
        """Perform quality checks on files. One file at a time."""
        for day in np.arange(self.timePeriod[0], self.timePeriod[1], np.timedelta64(1, 'D')):
            try:
                ds, fileOut = self.loadFiles(day)
            except IndexError:
                continue
            except OSError as e:
                if "Unknown file format" in str(e):
                    continue
                else:
                    raise
            ds = self.checkAngle(ds)
            ds = self.computeRainFlag(ds)
            ds = self.checkForSun(ds)
            ds = self.radomeMonitoring(ds, day)
            ds = self.updateRadomeStatus(ds, day)
            ds = self.combineFlags(ds)
            self.saveFile(ds, day, fileOut)

    def mergeMeasurementRetrieval(self, ds: xr.Dataset, dsR: Optional[xr.Dataset]) -> xr.Dataset:
        """Merge retrieved tb into ds."""
        try:
            if self.formatting == 'hatpro':
                ds['tb_retrieval'] = dsR['TBs']
                return ds
            _, index = np.unique(ds['time'], return_index=True)
            ds = ds.isel(time=index)
            _, index = np.unique(dsR['time'], return_index=True)
            dsRetrieval = dsR.isel(time=index)
            timeIndex = ds['time'].values
            dsRetrieval = dsRetrieval.reindex({'time': timeIndex}, method="nearest")
            dsRetrieval = dsRetrieval.rename({'number_frequencies': self.nfreq})
            ds['tb_retrieval'] = dsRetrieval['TBs']
            return ds
        except TypeError:
            if dsR is None:
                return ds
            else:
                raise

    @staticmethod
    def getClosestNFreq(freq: xr.DataArray, dsR: Optional[xr.Dataset]) -> Optional[xr.Dataset]:
        """Select the retrieved tb closes to the given frequencies."""
        try:
            aF = dsR['Freq']
            closestNAF = np.array([])
            for frequency in freq:
                cN = dsR['number_frequencies'][dsR['Freq'] ==
                                               functions.findNearest(aF, frequency).values]
                closestNAF = np.append(closestNAF, cN)
            closestNAF = closestNAF.astype(int)
            return dsR.sel(number_frequencies=closestNAF)
        except TypeError:
            if dsR is None:
                return dsR
            else:
                raise

    def loadFiles(self, time: np.datetime64) -> Tuple[xr.Dataset, str]:
        """Load files on and around position in file list."""
        times = [time - np.timedelta64(1, "D"), time, time + np.timedelta64(1, "D")]
        files = []
        rFiles = []
        for t in times:
            time_dt = t.astype("datetime64[m]").astype(dt.datetime)
            fnstructure = self.pathOrig.format(time_dt)
            filesIn = glob.glob(fnstructure)
            files.extend(filesIn)
            rFilesIn = glob.glob(self.pathRetrieval.format(time_dt))
            rFiles.extend(rFilesIn)
            if t == time:
                fileOut = self.pathQCDone.format(time_dt) + "/" + os.path.basename(filesIn[0])
                rFileExists = bool(rFilesIn)
        j = 1
        for i in np.arange(len(files)):
            try:
                ds = xr.open_dataset(files[i])
            except OSError as e:
                if j == len(files):
                    raise
                elif "Unknown file format" in str(e):
                    j+=1
                    continue
                else:
                    raise
        if rFileExists:
            try:
                dsR = xr.open_dataset(rFiles[0])
                for i in np.arange(len(rFiles) - 1):
                    dsRTemp = xr.open_dataset(rFiles[i + 1])
                    try:
                        dsR = xr.concat([dsR, dsRTemp], dim='time')
                    except ValueError as e:
                        if "is not present in all datasets" in str(e):
                            if "TBs" in list(dsRTemp.keys()):
                                dsR = dsRTemp
                            else:
                                dsRTemp = xr.open_dataset(rFiles[0])
                        elif "cannot be aligned because they have different dimension sizes" in str(e):
                            # In this case the number of frequencies calculated was switched to 14 or back to 100.
                            if i == 0:
                                dsR = dsRTemp
                            else:
                                dsRTemp = xr.open_dataset(rFiles[0])
                        else:
                            raise
                try:
                    dsR = functions.dropAdditionalTimeDims(dsR, dsRTemp)
                except NameError:
                    pass
            except IndexError:
                dsR = None
        else:
            dsR = None
        for i in np.arange(len(files) - j):
            try:
                dsTemp = xr.open_dataset(files[i + 1])
            except OSError as e:
                if "Unknown file format" in str(e):
                    continue
                else:
                    raise
            ds = xr.concat([ds, dsTemp], dim='time')
        try:
            ds = functions.dropAdditionalTimeDims(ds, dsTemp)
        except NameError:
            pass
        ds = self.prepareData(ds, dsR)
        return ds, fileOut

    def prepareData(self, ds: xr.Dataset, dsR: Optional[xr.Dataset]) -> xr.Dataset:
        """Load config files, and set up variables"""
        keys = list(ds.keys())
        if 'Freq' in keys:
            self.formatting = 'hatpro'
            frequencies = ds['Freq']
            self.freq = 'Freq'
            self.nfreq = 'number_frequencies'
            self.tbVar = 'TBs'
            dsR = self.getClosestNFreq(frequencies, dsR)
            ds = self.mergeMeasurementRetrieval(ds, dsR)
        elif 'freq_sb' in keys:
            self.formatting = 'hdcp2'
            frequencies = ds['freq_sb']
            self.freq = 'freq_sb'
            self.nfreq = 'n_freq'
            self.tbVar = 'tb'
            self.rainFlagVar = 'flag'
            self.sunFlagVar = 'flag'
            self.flagVar = "flag"
            self.rainBit = 3
            self.sunBit = 6
            dsR = self.getClosestNFreq(frequencies, dsR)
            ds = self.mergeMeasurementRetrieval(ds, dsR)
        try:
            ds['difference'] = ds[self.tbVar].copy()
            ds['difference'] = (["time", self.nfreq], ds[self.tbVar].values - ds['tb_retrieval'].values)
        except KeyError:
            pass  # No retrieval found
        ds = self.addRadomeVariablesToDataset(ds)
        return ds

    def addRadomeVariablesToDataset(self, ds: xr.Dataset) -> xr.Dataset:
        durationOfRainAttrs = OrderedDict([('long_name', 'duration of rain event'),
                                ('unit', 's'),
                                ('comment', "Consecutive rain events, without complete drying of radome are counted" +
                                 " as one.")
                                ])
        timeToDryAttrs = OrderedDict([('long_name', 'duration of wet radome after rain event'),
                                ('unit', 's'),
                                ('comment', "This variable is calculated only using the bias between spectral " +
                                 "retrieval and observation. It is the time between end of rain event (rain sensor)" +
                                 " and first time the difference is below a threshold.")
                                             ])
        maxDiffAttrs = OrderedDict([('long_name', 'Maximum difference between observation and spectral retrieval during and after rain event'),
            ('unit', 'K'),
            ('comments', "Only filled if it applies.")
            ])
        intDiffAttrs = OrderedDict([
            ('long_name', 'Integrated difference between observation and spectral retrieval during and after rain event'),
            ('unit', 'K s'),
            ('comments', "Only filled if it applies.")
            ])
        ds["durationOfRain"] = xr.Variable(dims=('time'),
                                           data=ds[self.rainFlagVar].values.copy(),
                                           encoding=dict(dtype="float32"),
                                           attrs = durationOfRainAttrs)
        ds["durationOfRain"][:] = np.nan
        ds["timeToDry"] = xr.Variable(dims=('time'),
                                      data=ds[self.rainFlagVar].values.copy(),
                                      encoding=dict(dtype="float32"),
                                      attrs=timeToDryAttrs
                                      )
        ds["timeToDry"][:] = np.nan
        ds["maxDiff"] = xr.Variable(dims=('time'),
                                           data=ds[self.rainFlagVar].values.copy(),
                                           encoding=dict(dtype="float32"),
                                      attrs=maxDiffAttrs
                                           )
        ds["maxDiff"][:] = np.nan
        ds["integratedDiff"] = xr.Variable(dims=('time'),
                                      data=ds[self.rainFlagVar].values.copy(),
                                      encoding=dict(dtype="float32"),
                                      attrs=intDiffAttrs
                                      )
        ds["integratedDiff"][:] = np.nan
        return ds

    def checkAngle(self, ds: xr.Dataset) -> xr.Dataset:
        """Check if angle is zenith or close"""
        zenith = 90
        tolerance = 1
        ds["ele_flag"] = ds[self.flagVar].copy()
        ds["ele_flag"] = xr.where((xr.where(ds['ele'] > zenith - tolerance, 0, 1) +
                                   xr.where(ds['ele'] < zenith + tolerance, 0, 1)) == 0, 0, 1)
        return ds

    def computeRainFlag(self, ds: xr.Dataset) -> xr.Dataset:
        """Compute where flag is set to rain."""
        if self.formatting == "hatpro":
            ds["rain_flag"] = ds["RF"].copy()
            return ds
        arrayRF = ds[self.rainFlagVar].values.copy()
        attrs_rf = OrderedDict([('long_name', 'rain flag'),
                                ('flag_masks', '[1]'),
                                ('flag_meanings', 'rain_flag'),
                                ('comments', "Extracted from 'flags'.")
                                ])
        for i in np.arange(len(arrayRF)):
            if np.isnan(arrayRF[i]):
                arrayRF[i] = 0
            elif arrayRF[i] < 2 ** self.rainBit:
                arrayRF[i] = 0
            else:
                arrayRF[i] = int(np.binary_repr(int(arrayRF[i]))[-self.rainBit - 1])
        ds['rain_flag'] = xr.Variable(dims=('time'),
                                      data=arrayRF.astype('float32'),
                                      attrs=attrs_rf,
                                      encoding=dict(dtype='float32')
                                      )
        return ds

    def checkForSun(self, ds: xr.Dataset) -> xr.Dataset:
        """Retrieve sun flag from dataset."""
        if self.formatting == 'hatpro':
            # Not creating sun flag for data in hatpro file format.
            return ds
        arraySF = ds[self.sunFlagVar].values.copy()
        attrs_sf = OrderedDict([('long_name', 'sun flag'),
                                ('flag_masks', '[1]'),
                                ('flag_meanings', 'sun_flag'),
                                ('comments', "Extracted from 'flags'.")
                                ])
        for i in np.arange(len(arraySF)):
            if np.isnan(arraySF[i]):
                arraySF[i] = 0
            elif arraySF[i] < 2 ** self.sunBit:
                arraySF[i] = 0
            else:
                arraySF[i] = int(np.binary_repr(int(arraySF[i]))[-self.sunBit - 1])
        ds['sun_flag'] = xr.Variable(dims=('time'),
                                     data=arraySF.astype('float32'),
                                     attrs=attrs_sf,
                                     encoding=dict(dtype='float32')
                                     )
        return ds

    def radomeMonitoring(self, ds: xr.Dataset, day: np.datetime64) -> xr.Dataset:
        """Add new data to radome monitoring netCDF"""
        try:
            ds = self.qcRetrieval(ds)
            ds = self.computeTimeToDry(ds)
            self.saveRadomeMonitoringTimeSeries(ds, day)
            self.saveRadomeMonitoringPlot(day)
        except KeyError:
            # No retrieval data found.
            pass
        return ds

    def qcRetrieval(self, ds: xr.Dataset) -> xr.Dataset:
        """Check if retrieval is a flat line and replace with np.nan."""
        if self.formatting == "e-profile":
            freq = self.freq
        else:
            freq = self.nfreq
        channel = ds[freq].values[-4]
        ds["tb_retrieval_10min_std"] = ds["tb_retrieval"].loc[{freq: channel}].copy()
        df_avg = ds['tb_retrieval'].loc[{freq: channel}].to_pandas().to_frame().rename(columns={0: self.tbVar})
        df_avg.index.rename('time', inplace=True)
        df_avg = df_avg.rolling("10min").std()
        df_avg.index = df_avg.index - np.timedelta64(5, "m")
        da_avg = xr.Dataset(df_avg)[self.tbVar]
        ds["tb_retrieval_10min_std"] = da_avg.reindex({"time": ds["time"]}, method="nearest")
        ds['q_retrieval'] = ds['rain_flag'].copy()
        ds['q_retrieval'][:] = 0
        if np.any(ds["tb_retrieval_10min_std"].values < 0.01):
            for t in ds['time'][::60]:
                if ds["tb_retrieval_10min_std"].loc[{'time': t}] < 0.01:
                    ds['q_retrieval'].loc[{'time': slice(t, t + np.timedelta64(20, 'm'))}] = 1
                    ds['difference'].loc[{'time': slice(t, t + np.timedelta64(20, 'm'))}] = np.nan
                    ds['tb_retrieval'].loc[{'time': slice(t, t + np.timedelta64(20, 'm'))}] = np.nan
        return ds

    def computeTimeToDry(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate the time it takes for the radome to dry after the end of a rain event."""
        wetRadomeTimeSeries = []
        dry = True
        endOfRain = None
        firstTimeBelowThreshold = None
        if self.formatting == "e-profile":
            threshold = nanmedian(ds['difference'].loc[{self.freq: 53.86}].values[ds["rain_flag"] == 0]) + 2
            ds['difference'].loc[{self.freq: 53.86}][-1] = threshold - 2
            difference = ds['difference'].loc[{self.freq: 53.86}]
        else:
            threshold = nanmedian(ds['difference'].loc[{self.nfreq: 9}].values[ds["rain_flag"] == 0]) + 2
            ds['difference'].loc[{self.nfreq: 9}][-1] = threshold - 2
            difference = ds['difference'].loc[{self.nfreq: 9}]
        radomeWetCondition = (difference > threshold) & (ds['ele_flag'] == 0) & (ds['q_retrieval'] == 0)
        buffer_time_func = lambda diff: np.timedelta64(180, "s") * diff
        tm1 = ds["time"][0] - np.timedelta64(1, "s")
        lastDiff = 0
        integratedDiff = 0
        maxDiff = 0
        firstMissing = None
        for time in ds['time']:
            if np.isnan(difference.sel(time=time)):
                # skip missing values
                wetRadomeTimeSeries.append(0)
                if firstMissing is None:
                    firstMissing = time
                continue
            rain = ds['rain_flag'].sel(time=time) == 1
            if dry and not rain:
                wetRadomeTimeSeries.append(0)
                continue
            radomeWet = radomeWetCondition.loc[{'time': time}]
            wetRadomeTimeSeries.append(int(not dry))
            deltaT = time - tm1
            integratedDiff += ((lastDiff + difference.loc[{'time': time}]) / 2
                               ) * (deltaT / np.timedelta64(1, 's'))
            lastDiff = difference.loc[{'time': time}]
            maxDiff = np.max([maxDiff, lastDiff])
            tm1 = time
            if rain and dry:
                if radomeWet:
                    dry = False
                    beginOfRain = time
            elif not (rain or dry):  # wet and rain stopped
                if endOfRain is None:  # first time stamp after end of rain
                    endOfRain = time
                    durationOfRain = endOfRain - beginOfRain
                if radomeWet:  # radome is still wet
                    firstMissing = None  # still wet after data gap
                    continue
                elif firstTimeBelowThreshold is None:  # radome is dry for the first time
                    firstTimeBelowThreshold = time
                    buffer_time = buffer_time_func(difference.loc[{'time': time}].values)
                elif firstTimeBelowThreshold + buffer_time > time:  # add 2 minutes of flagged data after dry
                    continue
                else:
                    dry = True
                    try:
                        # if dried during large data gap, shortest possible dry time is used
                        endTimeOfDrying = np.nanmin([firstTimeBelowThreshold.values, firstMissing.values])
                    except (TypeError, AttributeError):
                        # no data gap during drying event
                        endTimeOfDrying = firstTimeBelowThreshold
                    timeToDry = endTimeOfDrying - endOfRain
                    ds['durationOfRain'].loc[{'time': firstTimeBelowThreshold}] = durationOfRain / np.timedelta64(1,
                                                                                                                  's')
                    ds['timeToDry'].loc[{'time': firstTimeBelowThreshold}] = timeToDry / np.timedelta64(1, 's')
                    ds['maxDiff'].loc[{'time': firstTimeBelowThreshold}] = maxDiff
                    ds['integratedDiff'].loc[{'time': firstTimeBelowThreshold}] = integratedDiff
                    firstTimeBelowThreshold = None
                    integratedDiff = 0
                    maxDiff = 0
                    endOfRain = None
            elif not dry and rain:  # rain starts again before radome is dry
                if endOfRain is not None:
                    # Likely rain continued but was not detected by rain sensor
                    endOfRain = None
                    firstTimeBelowThreshold = None
                    firstMissing = None
        ds["radome_wet"] = xr.Variable(dims=('time'),
                                            data=np.array(wetRadomeTimeSeries).astype('float32'),
                                            encoding=dict(dtype='float32')
                                            )
        return ds

    def saveRadomeMonitoringTimeSeries(self, ds: xr.Dataset, day: np.datetime64):
        """Save a dataset which only contains the hourly maximum of the time to dry and rain duration."""
        fileMonth = self.pathRadomeMonitoring + "/" + self.radomeMonthFname.format(day.astype("datetime64[m]").astype(dt.datetime))
        dsRadomeDay = ds.copy(deep = True)
        keys = list(dsRadomeDay.keys())
        keep_keys = ["durationOfRain", "timeToDry"]
        drop_keys = [i for i in keys if not (i in keep_keys)]
        dsRadomeDay = dsRadomeDay.drop_vars(drop_keys)
        dsRadomeDay = dsRadomeDay.reindex({"time": ds["time"][~np.isnan(ds["durationOfRain"])]})
        dsRadomeDay = dsRadomeDay.loc[{"time": slice(day, day + np.timedelta64(1, "D"))}]
        try:
            dsRadomeMonth_t = xr.open_dataset(fileMonth)
            dsRadomeMonth = xr.concat([dsRadomeMonth_t, dsRadomeDay], dim = "time")
            dsRadomeMonth_t.close()
        except FileNotFoundError:
            dsRadomeMonth = dsRadomeDay
        except ValueError:
            dsRadomeMonth = dsRadomeDay
            os.remove(fileMonth)
        _, index = np.unique(dsRadomeMonth['time'], return_index=True)
        dsRadomeMonth = dsRadomeMonth.isel({"time": index})
        for coord in list(dsRadomeMonth.coords):
            if "time" in coord:
                dsRadomeMonth[coord].encoding['units'] = 'seconds since 1970-01-01 00:00:00 UTC'
        if len(dsRadomeMonth["time"].values) > 0:
            functions.createDir(os.path.dirname(fileMonth))
            dsRadomeMonth.to_netcdf(fileMonth)
        dsRadomeMonth.close()

    def saveRadomeMonitoringPlot(self, day):
        """Save a standard visualization of the evolution of time to dry since last radome change."""
        # load all radome monitoring files
        files = glob.glob(self.pathRadomeMonitoring + "wet_radome_*.nc")
        try:
            ds = xr.open_dataset(files[0])
        except IndexError:
            return
        for file in files[1:]:
            ds_temp = xr.open_dataset(file)
            ds = xr.merge([ds, ds_temp])
        # make plot
        locator_day = mdates.DayLocator(interval=1)
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_axes([0, 0, 1, 1])
        time = "time"
        ttd = 'timeToDry'
        duration = 'durationOfRain'
        x = ds[time].values
        ax.scatter(x, ds[duration].values / 60, zorder=2, label='duration of rain', s=8)
        ax.bar(x, ds[ttd].values / 60, width=1.2, bottom=None, align='center', label='time to dry')
        ax.axhline(10, label="service required", c="red")
        ax.axhline(3, label="warning", c="orange")
        ax.set_xlabel('Date', fontsize=16)
        ax.set_ylabel('Duration in minutes', fontsize=16)
        ax.set_ylim(0, 60)
        if self.currentRadome < np.datetime64("1971-01-01"):
            xlimLow = ds["time"].values[0]
        else:
            xlimLow = self.currentRadome
        ax.set_xlim(xlimLow, day + np.timedelta64(1, "D"))
        ax.set_title('Radome Hygroscopic Properties', fontsize=18)
        ax.xaxis.set_minor_locator(locator_day)
        ax.legend()
        fname = "radome_status_{:%Y%m%d.png}".format(self.currentRadome.astype("datetime64[s]").astype(dt.datetime))
        fig.savefig(self.pathRadomeMonitoring + fname, bbox_inches='tight')
        plt.close(fig)

    def updateRadomeStatus(self, ds: xr.Dataset, day: np.datetime64) -> xr.Dataset:
        """Using time to dry add a status flag for the radome quality [0,1,2] good acceptable bad"""
        # This is called apon iteratively for every day
        # Limit for time to dry for radomeStatusFlags
        qLimitsRadome = [3, 10]  # minutes
        # Extract time of last radome switch
        try:
            newRadomeTimes = self.newRadomeTimes
        except KeyError:
            newRadomeTimes = []
        if newRadomeTimes:
            #determine newest radome (time of installation) for a given day
            newestRadome = np.datetime64("1970-01-01") + np.timedelta64(1, "D")  # np.datetime64(newRadomeTimes[0])
            for time in newRadomeTimes:
                if (np.datetime64(time) > newestRadome) and (np.datetime64(time) < day):
                    newestRadome = np.datetime64(time)
        else:
            # no Radome Times defined in the station Metadata
            if self.currentRadome == np.datetime64("1970-01-01"):
                # artificially add a newer radome if its still the original one to start the counting
                newestRadome = self.currentRadome + np.timedelta64(1, "D")
            else:
                # the obove was done in a prior iteration
                newestRadome = self.currentRadome
        # retrieve max time to dry
        if self.currentRadome < newestRadome:
            # accounting for switch of radome
            self.radomeStatus = 0
            self.currentRadome = newestRadome
            step = np.timedelta64(1, 'M')
            months = np.arange(newestRadome.astype('datetime64[M]'), day.astype('datetime64[M]') + step, step)
            self.maxTimeToDry = 0
            for month in months:
                fileMonth = self.pathRadomeMonitoring + self.radomeMonthFname.format(month.astype(dt.datetime))
                try:
                    dsRadomeMonth = xr.open_dataset(fileMonth)
                    sliceStart = max(newestRadome, dsRadomeMonth["time"].values[0])
                    sliceEnd = min(day + np.timedelta64(1, "D"), dsRadomeMonth["time"].values[-1])
                    maxTimeToDryMonth = nanmax(dsRadomeMonth["timeToDry"].loc[
                        {"time": slice(sliceStart, sliceEnd)}].values)
                    if self.maxTimeToDry < maxTimeToDryMonth:
                        self.maxTimeToDry = maxTimeToDryMonth
                except FileNotFoundError:
                    pass
                except ValueError:
                    if sliceStart >= sliceEnd:
                        pass
                    else:
                        raise
        else:
            # is only accessed if this is not the first day in time period, i.e. self.maxTimeToDry is already defined.
            try:
                maxTimeToDryDay = nanmax(ds["timeToDry"].loc[{"time": slice(day, day + np.timedelta64(1, "D"))}].values)
                self.maxTimeToDry = nanmax([self.maxTimeToDry, maxTimeToDryDay])
            except KeyError:
                pass
            except ValueError as e:
                if "a.size==0" in str(e):
                    pass
                else:
                    raise
        # derive radome status from max time to dry
        radomeStatusT = 0
        for i in np.arange(2):
            if qLimitsRadome[i] < self.maxTimeToDry/60:
                radomeStatusT = i + 1
        if radomeStatusT > self.radomeStatus:
            self.radomeStatus = radomeStatusT
        # add radome status as a variable to ds
        arrayRS = ds[self.rainFlagVar].values.copy()
        arrayRS[:] = self.radomeStatus
        attrs_rs = OrderedDict([('long_name', 'radome_status'),
                                ('flag_masks', '[0,1,2]'),
                                ('flag_meanings', 'good, acceptable, bad'),
                                ('comments', "Derived from the max time to dry after a rain event. 1: more than 3 " +
                                 "minutes, 2: more than 10 minutes.")
                                ])
        ds['radome_status'] = xr.Variable(dims=('time'),
                                          data=arrayRS.astype('float32'),
                                          attrs=attrs_rs,
                                          encoding=dict(dtype='float32')
                                          )
        return ds

    def fillEmptyFlags(self, attrs_flag: dict, firstAdditionalFlag: int) -> dict:
        """Fill empty flag slots up to firstAdditionalFlag and remove zero if present."""
        noOfBits = len(np.binary_repr(int(firstAdditionalFlag/2)))
        try:
            flag_masks = list(attrs_flag["flag_masks"])
            standardFM = True
        except KeyError:
            flag_masks = list(attrs_flag["flag_values"])
            standardFM = False
        flag_meanings = attrs_flag["flag_meanings"].split()
        for i in np.arange(noOfBits):
            bit = 2 ** i
            try:
                if flag_masks[i] == 0:
                    flag_masks.pop(i)
                    flag_meanings.pop(i)
                if flag_masks[i] == bit:
                    continue
                else:
                    flag_masks.insert(i, bit)
                    flag_meanings.insert(i, "_")
            except IndexError:
                flag_masks.append(bit)
                flag_meanings.append("_")
        flag_meanings_text = ""
        for flag_meaning in flag_meanings:
            flag_meanings_text = flag_meanings_text + flag_meaning + " "
        flag_meanings_text = flag_meanings_text[:-1]
        if standardFM:
            attrs_flag["flag_masks"] = np.array(flag_masks)
        else:
            attrs_flag["flag_values"] = np.array(flag_masks)
        attrs_flag["flag_meanings"] = flag_meanings_text
        return attrs_flag

    def combineFlags(self, ds: xr.Dataset) -> xr.Dataset:
        """Combine flags to one."""
        try:
            attrs_flag = ds["flag"].attrs
        except KeyError:
            # Quality control flags cannot be added to Hatpro file format. Use HDCP2 file format.
            return ds
        attrs_flag["long_name"] = 'quality control flags'
        additional_flag_masks = np.array([1024])
        attrs_flag = self.fillEmptyFlags(attrs_flag, additional_flag_masks[0])
        try:
            attrs_flag["flag_masks"] = np.append(attrs_flag["flag_masks"], additional_flag_masks)
        except KeyError:
            attrs_flag["flag_values"] = np.append(attrs_flag["flag_values"], additional_flag_masks)
            attrs_flag = OrderedDict([('flag_masks', v) if k == 'flag_values' else (k, v) for k, v in attrs_flag.items()])
        try:
            attrs_flag["flag_meanings"] = attrs_flag["flag_meanings"] + " radome_wet"
        except KeyError:
            raise
        additional_comment = " radome is assumed wet when there is a strong deviation in tb from what is expected;"
        try:
            attrs_flag["comment"] = attrs_flag["comment"] + additional_comment
        except KeyError:
            attrs_flag["comment"] = additional_comment
        flag = ds['flag'].values
        additional_flag_names = ["radome_wet"]
        for i in np.arange(len(additional_flag_names)):
            try:
                afn = additional_flag_names[i]
                flag = nansum(np.dstack((flag, ds[afn].values * additional_flag_masks[i])), 2).flatten()
            except KeyError:
                pass
        ds['flag'] = xr.Variable(dims=('time'),
                                 data=flag.astype('float32'),
                                 attrs=attrs_flag,
                                 encoding=dict(dtype='float32')
                                 )
        return ds

    def saveFile(self, ds: xr.Dataset, day: np.datetime64, fileOut: str, ele: bool = False):
        """Remove all unwanted variables, cut out one day, save to disk."""
        varsToDrop = ["ele_flag", 'sun_flag', 'rain_flag', 'durationOfRain', 'timeToDry',
                      'q_retrieval', 'difference', "tb_retrieval", "zero_tb_retrieval", "radome_wet",
                      "zero_durationOfRain", "zero_timeToDry", "flag_binary", "Nbit", "tb_retrieval_10min_std"]
        ds = ds.drop_vars(varsToDrop, errors="ignore")
        ds = ds.loc[{'time': slice(day, day + np.timedelta64(1, 'D') - np.timedelta64(0, 'ns'))}]
        functions.createDir(os.path.dirname(fileOut))
        for coord in list(ds.coords):
            if "time" in coord:
                ds[coord].encoding['units'] = 'seconds since 1970-01-01 00:00:00 UTC'

        if self.formatting == "e-profile":
            fileOut = fileOut[:-7] + "0000" + fileOut[-3:]
        else:
            fileOut = fileOut[:-9] + "000000" + fileOut[-3:]
        ds.to_netcdf(fileOut)


class MWRQualityEprof(MWRQuality):
    """Check the data quality of MWR data for time period (E-profile data format)."""

    def __init__(self, ph: ProcessHandler):
        """Run pipe"""
        super(MWRQualityEprof, self).__init__(ph)

    def runQC(self):
        """Perform quality checks on files. One file at a time."""
        for day in np.arange(self.timePeriod[0], self.timePeriod[1], np.timedelta64(1, 'D')):
            try:
                ds, fileOut = self.loadFiles(day)
            except IndexError:
                continue
            except OSError as e:
                if "Unknown file format" in str(e):
                    continue
                else:
                    raise
            ds = self.checkAngle(ds)
            ds = self.computeRainFlag(ds)
            ds = self.checkForSun(ds)
            ds = self.radomeMonitoring(ds, day)
            ds = self.updateRadomeStatus(ds, day)
            ds = self.combineFlags(ds)
            self.saveFile(ds, day, fileOut)

    def mergeMeasurementRetrieval(self, ds: xr.Dataset, dsR: Optional[xr.Dataset]) -> xr.Dataset:
        """Merge retrieved tb into ds."""
        try:
            if self.formatting == 'hatpro':
                ds['tb_retrieval'] = dsR['TBs']
                return ds
            # make sure ds and dsR have unique time stamps
            _, index = np.unique(ds['time'], return_index=True)
            ds = ds.isel(time=index)
            _, index = np.unique(dsR['time'], return_index=True)
            dsRetrieval = dsR.isel(time=index)
            timeIndex = ds['time'].values
            dsRetrieval = dsRetrieval.reindex({'time': timeIndex}, method="nearest")
            dsRetrieval = dsRetrieval.rename({'number_frequencies': self.freq})
            dsRetrieval[self.freq] = ds[self.freq]
            ds['tb_retrieval'] = dsRetrieval['TBs']
            return ds
        except TypeError:
            if dsR is None:
                return ds
            else:
                raise

    def prepareData(self, ds: xr.Dataset, dsR: Optional[xr.Dataset]) -> xr.Dataset:
        """Load config files, and set up variables"""
        keys = list(ds.keys())
        self.formatting = 'e-profile'
        self.freq = 'frequency'
        frequencies = ds[self.freq]
        self.tbVar = 'tb'
        self.rainFlagVar = 'quality_flag'
        self.sunFlagVar = 'quality_flag'
        self.flagVar = "quality_flag"
        self.rainBit = 5
        self.sunBit = 6
        dsR = self.getClosestNFreq(frequencies, dsR)
        ds = self.mergeMeasurementRetrieval(ds, dsR)
        try:
            ds['difference'] = ds[self.tbVar].copy()
            ds['difference'] = (["time", self.freq], ds[self.tbVar].values - ds['tb_retrieval'].values)
        except KeyError:
            pass  # No retrieval found
        ds = ds.rename({self.flagVar: "quality_flag_orig"})
        ds[self.flagVar] = ds["quality_flag_orig"].loc[{self.freq: ds[self.freq].values[0]}]
        ds = self.addRadomeVariablesToDataset(ds)
        return ds

    def combineFlags(self, ds: xr.Dataset) -> xr.Dataset:
        """Combine flags to one."""
        ds = ds.drop_vars(self.flagVar)
        ds = ds.rename({"quality_flag_orig": self.flagVar})
        for freq in ds[self.freq]:
            # make binary flag array
            maxBit = np.max(ds["quality_flag"].attrs["flag_masks"])
            binArr = functions.createBinaryArray(ds["quality_flag"].loc[{self.freq: freq}].values, maxBit)
            noOfBits = len(np.binary_repr(int(maxBit)))
            for b in np.arange(1, noOfBits + 1):
                binArr[:, -b] = binArr[:, -b] / b
            # add newly computed flags to old ones
            for flag, bit in zip(["missing_data", "radome_wet"], [0, 5]):
                # add missing_data to missing and radome_wet to rain
                try:
                    binArr[:, noOfBits - bit - 1] = np.nanmax([ds[flag].values, binArr[:, noOfBits - bit - 1]], axis=0)
                except KeyError:
                    pass
            # undo the binary array thing
            flag_sum = binArr[:, noOfBits - 1].copy()
            for bit in np.arange(1, noOfBits):
                flag_sum = flag_sum + binArr[:, noOfBits - bit - 1] * 2 ** bit
            # replace flag for this freq
            ds["quality_flag"].loc[{self.freq: freq}] = flag_sum
        return ds


if __name__ == '__main__':
    try:
        station = sys.argv[1]
    except IndexError:
        sys.exit("Station must be defined in call.")
    ph = ProcessHandler(pid = station, stations = station)
    try:
        format = functions.getEnvVars(directory=ph.envVarsDir, names="general")["format"]
        if format == "e-profile":
            MWRQualityEprof(ph)
        else:
            MWRQuality(ph)
    except KeyError:
        MWRQuality(ph)

