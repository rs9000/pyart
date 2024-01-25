"""
Utilities for reading Datamet files.

"""

import datetime
import numpy as np
from matplotlib import pyplot as plt

import pyart.retrieve
from pyart.config import FileMetadata
from pyart.core.radar import Radar
from pyart.io.common import make_time_unit_str
import xradar as xd
from pyart.testing import get_test_data

def read_datamet_xradar(filename, **kwargs):
    """
    Read a Datamet file.

    Parameters
    ----------
    filename : str
        Name of Datamet path to read data from.

    Returns
    -------
    radar : Radar
        Radar object.

    """

    # create metadata retrieval object
    filemetadata = FileMetadata("datamet")

    # Open Datamet file
    dtree = xd.io.open_datamet_datatree(filename)

    # check the number of slices
    nslices = len(dtree.groups) - 1

    # latitude, longitude and altitude
    latitude = filemetadata("latitude")
    longitude = filemetadata("longitude")
    altitude = filemetadata("altitude")
    latitude["data"] = np.array([dtree.variables['latitude']])
    longitude["data"] = np.array([dtree.variables['longitude']])
    altitude["data"] = np.array([dtree.variables['altitude']])

    # sweep_number (is the sweep index)
    sweep_number = filemetadata("sweep_number")
    sweep_number["data"] = np.arange(nslices, dtype="int32")

    rays_per_sweep = np.empty(nslices, dtype="int32")
    nbins_sweep = np.empty(nslices, dtype="int32")
    alist = []
    elist = []
    tlist = []
    for i in range(nslices):
        rays_per_sweep[i] = len(dtree[f"sweep_{i}"].azimuth)
        nbins_sweep[i] = len(dtree[f"sweep_{i}"].range)
        alist.extend(dtree[f"sweep_{i}"].azimuth.data)
        elist.extend(dtree[f"sweep_{i}"].elevation.data)
        tlist.extend((dtree[f"sweep_{i}"].time.data - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))

    # all sweeps have to have the same number of range bins
    if any(nbins_sweep != nbins_sweep[0]):
        print("Warn: Resolution must be the same for all sweeps")

    nbins = nbins_sweep[0]
    ssri = np.cumsum(np.append([0], rays_per_sweep[:-1])).astype("int32")
    seri = np.cumsum(rays_per_sweep).astype("int32") - 1

    # total number of rays and sweep start ray index and end
    total_rays = sum(rays_per_sweep)
    sweep_start_ray_index = filemetadata("sweep_start_ray_index")
    sweep_end_ray_index = filemetadata("sweep_end_ray_index")
    sweep_start_ray_index["data"] = ssri
    sweep_end_ray_index["data"] = seri

    # azimuth
    azimuth = filemetadata("azimuth")
    azimuth["data"] = np.array(alist)

    # elevation
    elevation = filemetadata("elevation")
    elevation["data"] = np.array(elist)

    # time
    time = filemetadata("time")
    start_time = datetime.datetime.utcfromtimestamp(tlist[0])
    time["units"] = make_time_unit_str(start_time)
    time["data"] = tlist - tlist[0]

    # range
    _range = filemetadata("range")
    max_sweep = "sweep_" + str(np.argmax(nbins_sweep))
    _range["data"] = dtree[max_sweep].range.data
    _range["meters_to_center_of_first_gate"] = dtree[max_sweep].range.meters_to_center_of_first_gate
    _range["meters_between_gates"] = dtree[max_sweep].range.meters_between_gates

    # sweep_type
    scan_type = "ppi"

    # sweep_mode, fixed_angle
    sweep_mode = filemetadata("sweep_mode")
    fixed_angle = filemetadata("fixed_angle")
    sweep_mode["data"] = np.array(nslices * ["azimuth_surveillance"])

    # containers for data
    fixed_angle["data"] = np.empty(nslices, dtype="float64")

    # read data
    fields = {}
    max_sweep_dim = max(nbins_sweep)

    for i in range(nslices):
        fixed_angle["data"][i] = float(dtree[f"sweep_{i}"].elevation[0])
        for item in dtree[f"sweep_{i}"].data_vars:
            if item in xd.io.backends.datamet.xradar_mapping.keys():
                key = xd.io.backends.datamet.xradar_mapping[item]
                if key not in fields.keys():
                    fields[key] = filemetadata(item)
                    fields[key]["units"] = dtree[f"sweep_{i}"][item].attrs.get("units", "")
                    fields[key]["_FillValue"] = dtree[f"sweep_{i}"][item].attrs.get("_FillValue", -1)
                    fields[key]["standard_name"] = dtree[f"sweep_{i}"][item].attrs.get("standard_name", "")

                    fields_item_data = np.empty((rays_per_sweep[0], max_sweep_dim))
                    fields_item_data[:] = np.nan
                    sweep_data = dtree[f"sweep_{i}"][item].data
                    fields_item_data[0:sweep_data.shape[0], 0:sweep_data.shape[1]] = sweep_data
                    fields[key]["data"] = fields_item_data
                else:
                    fields_item_data = np.empty((rays_per_sweep[0], max_sweep_dim))
                    fields_item_data[:] = np.nan
                    sweep_data = dtree[f"sweep_{i}"][item].data
                    fields_item_data[0:sweep_data.shape[0], 0:sweep_data.shape[1]] = sweep_data
                    fields[key]["data"] = np.append(fields[key]["data"], fields_item_data, axis=0)

    # metadata
    metadata = filemetadata("metadata")
    metadata["radar_name"] = dtree.origin
    metadata["conversion_software"] = "datamet"
    instrument_parameters = {}

    return Radar(
        time,
        _range,
        fields,
        metadata,
        scan_type,
        latitude,
        longitude,
        altitude,
        sweep_number,
        sweep_mode,
        fixed_angle,
        sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth,
        elevation,
        instrument_parameters=instrument_parameters,
    )


if __name__ == '__main__':
    radar = read_datamet_xradar(r'C:\RADAR\RAW\2023\11\22\0300\VOL\H\LAURO')
    fig = plt.figure(figsize=[10, 10])
    display = pyart.graph.RadarMapDisplay(radar)
    display.plot_ppi_map('reflectivity', sweep=10)
