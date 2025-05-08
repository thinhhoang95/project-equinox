Dataset Information:
Dimensions: {'valid_time': 24, 'latitude': 169, 'longitude': 225, 'pressure_level': 2}

Coordinates:
  - number: ()
  - valid_time: (24,)
  - latitude: (169,)
  - longitude: (225,)
  - expver: (24,)
  - pressure_level: (2,)

Data Variables:
  - u10
    Dimensions: ('valid_time', 'latitude', 'longitude')
    Shape: (24, 169, 225)
    Attributes: {'GRIB_paramId': np.int64(165), 'GRIB_dataType': 'an', 'GRIB_numberOfPoints': np.int64(38025), 'GRIB_typeOfLevel': 'surface', 'GRIB_stepUnits': np.int64(1), 'GRIB_stepType': 'instant', 'GRIB_gridType': 'regular_ll', 'GRIB_uvRelativeToGrid': np.int64(0), 'GRIB_NV': np.int64(0), 'GRIB_Nx': np.int64(225), 'GRIB_Ny': np.int64(169), 'GRIB_cfName': 'unknown', 'GRIB_cfVarName': 'u10', 'GRIB_gridDefinitionDescription': 'Latitude/Longitude Grid', 'GRIB_iDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_iScansNegatively': np.int64(0), 'GRIB_jDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_jPointsAreConsecutive': np.int64(0), 'GRIB_jScansPositively': np.int64(0), 'GRIB_latitudeOfFirstGridPointInDegrees': np.float64(72.0), 'GRIB_latitudeOfLastGridPointInDegrees': np.float64(30.0), 'GRIB_longitudeOfFirstGridPointInDegrees': np.float64(-15.0), 'GRIB_longitudeOfLastGridPointInDegrees': np.float64(41.0), 'GRIB_missingValue': np.float64(3.4028234663852886e+38), 'GRIB_name': '10 metre U wind component', 'GRIB_shortName': '10u', 'GRIB_totalNumber': np.int64(0), 'GRIB_units': 'm s**-1', 'long_name': '10 metre U wind component', 'units': 'm s**-1', 'standard_name': 'unknown', 'GRIB_surface': np.float64(0.0)}

  - v10
    Dimensions: ('valid_time', 'latitude', 'longitude')
    Shape: (24, 169, 225)
    Attributes: {'GRIB_paramId': np.int64(166), 'GRIB_dataType': 'an', 'GRIB_numberOfPoints': np.int64(38025), 'GRIB_typeOfLevel': 'surface', 'GRIB_stepUnits': np.int64(1), 'GRIB_stepType': 'instant', 'GRIB_gridType': 'regular_ll', 'GRIB_uvRelativeToGrid': np.int64(0), 'GRIB_NV': np.int64(0), 'GRIB_Nx': np.int64(225), 'GRIB_Ny': np.int64(169), 'GRIB_cfName': 'unknown', 'GRIB_cfVarName': 'v10', 'GRIB_gridDefinitionDescription': 'Latitude/Longitude Grid', 'GRIB_iDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_iScansNegatively': np.int64(0), 'GRIB_jDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_jPointsAreConsecutive': np.int64(0), 'GRIB_jScansPositively': np.int64(0), 'GRIB_latitudeOfFirstGridPointInDegrees': np.float64(72.0), 'GRIB_latitudeOfLastGridPointInDegrees': np.float64(30.0), 'GRIB_longitudeOfFirstGridPointInDegrees': np.float64(-15.0), 'GRIB_longitudeOfLastGridPointInDegrees': np.float64(41.0), 'GRIB_missingValue': np.float64(3.4028234663852886e+38), 'GRIB_name': '10 metre V wind component', 'GRIB_shortName': '10v', 'GRIB_totalNumber': np.int64(0), 'GRIB_units': 'm s**-1', 'long_name': '10 metre V wind component', 'units': 'm s**-1', 'standard_name': 'unknown', 'GRIB_surface': np.float64(0.0)}

  - cape
    Dimensions: ('valid_time', 'latitude', 'longitude')
    Shape: (24, 169, 225)
    Attributes: {'GRIB_paramId': np.int64(59), 'GRIB_dataType': 'an', 'GRIB_numberOfPoints': np.int64(38025), 'GRIB_typeOfLevel': 'surface', 'GRIB_stepUnits': np.int64(1), 'GRIB_stepType': 'instant', 'GRIB_gridType': 'regular_ll', 'GRIB_uvRelativeToGrid': np.int64(0), 'GRIB_NV': np.int64(0), 'GRIB_Nx': np.int64(225), 'GRIB_Ny': np.int64(169), 'GRIB_cfName': 'unknown', 'GRIB_cfVarName': 'cape', 'GRIB_gridDefinitionDescription': 'Latitude/Longitude Grid', 'GRIB_iDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_iScansNegatively': np.int64(0), 'GRIB_jDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_jPointsAreConsecutive': np.int64(0), 'GRIB_jScansPositively': np.int64(0), 'GRIB_latitudeOfFirstGridPointInDegrees': np.float64(72.0), 'GRIB_latitudeOfLastGridPointInDegrees': np.float64(30.0), 'GRIB_longitudeOfFirstGridPointInDegrees': np.float64(-15.0), 'GRIB_longitudeOfLastGridPointInDegrees': np.float64(41.0), 'GRIB_missingValue': np.float64(3.4028234663852886e+38), 'GRIB_name': 'Convective available potential energy', 'GRIB_shortName': 'cape', 'GRIB_totalNumber': np.int64(0), 'GRIB_units': 'J kg**-1', 'long_name': 'Convective available potential energy', 'units': 'J kg**-1', 'standard_name': 'unknown', 'GRIB_surface': np.float64(0.0)}

  - cin
    Dimensions: ('valid_time', 'latitude', 'longitude')
    Shape: (24, 169, 225)
    Attributes: {'GRIB_paramId': np.int64(228001), 'GRIB_dataType': 'fc', 'GRIB_numberOfPoints': np.int64(38025), 'GRIB_typeOfLevel': 'surface', 'GRIB_stepUnits': np.int64(1), 'GRIB_stepType': 'instant', 'GRIB_gridType': 'regular_ll', 'GRIB_uvRelativeToGrid': np.int64(0), 'GRIB_NV': np.int64(0), 'GRIB_Nx': np.int64(225), 'GRIB_Ny': np.int64(169), 'GRIB_cfName': 'unknown', 'GRIB_cfVarName': 'cin', 'GRIB_gridDefinitionDescription': 'Latitude/Longitude Grid', 'GRIB_iDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_iScansNegatively': np.int64(0), 'GRIB_jDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_jPointsAreConsecutive': np.int64(0), 'GRIB_jScansPositively': np.int64(0), 'GRIB_latitudeOfFirstGridPointInDegrees': np.float64(72.0), 'GRIB_latitudeOfLastGridPointInDegrees': np.float64(30.0), 'GRIB_longitudeOfFirstGridPointInDegrees': np.float64(-15.0), 'GRIB_longitudeOfLastGridPointInDegrees': np.float64(41.0), 'GRIB_missingValue': np.float64(3.4028234663852886e+38), 'GRIB_name': 'Convective inhibition', 'GRIB_shortName': 'cin', 'GRIB_totalNumber': np.int64(0), 'GRIB_units': 'J kg**-1', 'long_name': 'Convective inhibition', 'units': 'J kg**-1', 'standard_name': 'unknown', 'GRIB_surface': np.float64(0.0)}

  - hcc
    Dimensions: ('valid_time', 'latitude', 'longitude')
    Shape: (24, 169, 225)
    Attributes: {'GRIB_paramId': np.int64(188), 'GRIB_dataType': 'an', 'GRIB_numberOfPoints': np.int64(38025), 'GRIB_typeOfLevel': 'surface', 'GRIB_stepUnits': np.int64(1), 'GRIB_stepType': 'instant', 'GRIB_gridType': 'regular_ll', 'GRIB_uvRelativeToGrid': np.int64(0), 'GRIB_NV': np.int64(0), 'GRIB_Nx': np.int64(225), 'GRIB_Ny': np.int64(169), 'GRIB_cfName': 'unknown', 'GRIB_cfVarName': 'hcc', 'GRIB_gridDefinitionDescription': 'Latitude/Longitude Grid', 'GRIB_iDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_iScansNegatively': np.int64(0), 'GRIB_jDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_jPointsAreConsecutive': np.int64(0), 'GRIB_jScansPositively': np.int64(0), 'GRIB_latitudeOfFirstGridPointInDegrees': np.float64(72.0), 'GRIB_latitudeOfLastGridPointInDegrees': np.float64(30.0), 'GRIB_longitudeOfFirstGridPointInDegrees': np.float64(-15.0), 'GRIB_longitudeOfLastGridPointInDegrees': np.float64(41.0), 'GRIB_missingValue': np.float64(3.4028234663852886e+38), 'GRIB_name': 'High cloud cover', 'GRIB_shortName': 'hcc', 'GRIB_totalNumber': np.int64(0), 'GRIB_units': '(0 - 1)', 'long_name': 'High cloud cover', 'units': '(0 - 1)', 'standard_name': 'unknown', 'GRIB_surface': np.float64(0.0)}

  - u
    Dimensions: ('valid_time', 'pressure_level', 'latitude', 'longitude')
    Shape: (24, 2, 169, 225)
    Attributes: {'GRIB_paramId': np.int64(131), 'GRIB_dataType': 'an', 'GRIB_numberOfPoints': np.int64(38025), 'GRIB_typeOfLevel': 'isobaricInhPa', 'GRIB_stepUnits': np.int64(1), 'GRIB_stepType': 'instant', 'GRIB_gridType': 'regular_ll', 'GRIB_uvRelativeToGrid': np.int64(0), 'GRIB_NV': np.int64(0), 'GRIB_Nx': np.int64(225), 'GRIB_Ny': np.int64(169), 'GRIB_cfName': 'eastward_wind', 'GRIB_cfVarName': 'u', 'GRIB_gridDefinitionDescription': 'Latitude/Longitude Grid', 'GRIB_iDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_iScansNegatively': np.int64(0), 'GRIB_jDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_jPointsAreConsecutive': np.int64(0), 'GRIB_jScansPositively': np.int64(0), 'GRIB_latitudeOfFirstGridPointInDegrees': np.float64(72.0), 'GRIB_latitudeOfLastGridPointInDegrees': np.float64(30.0), 'GRIB_longitudeOfFirstGridPointInDegrees': np.float64(-15.0), 'GRIB_longitudeOfLastGridPointInDegrees': np.float64(41.0), 'GRIB_missingValue': np.float64(3.4028234663852886e+38), 'GRIB_name': 'U component of wind', 'GRIB_shortName': 'u', 'GRIB_totalNumber': np.int64(0), 'GRIB_units': 'm s**-1', 'long_name': 'U component of wind', 'units': 'm s**-1', 'standard_name': 'eastward_wind'}

  - v
    Dimensions: ('valid_time', 'pressure_level', 'latitude', 'longitude')
    Shape: (24, 2, 169, 225)
    Attributes: {'GRIB_paramId': np.int64(132), 'GRIB_dataType': 'an', 'GRIB_numberOfPoints': np.int64(38025), 'GRIB_typeOfLevel': 'isobaricInhPa', 'GRIB_stepUnits': np.int64(1), 'GRIB_stepType': 'instant', 'GRIB_gridType': 'regular_ll', 'GRIB_uvRelativeToGrid': np.int64(0), 'GRIB_NV': np.int64(0), 'GRIB_Nx': np.int64(225), 'GRIB_Ny': np.int64(169), 'GRIB_cfName': 'northward_wind', 'GRIB_cfVarName': 'v', 'GRIB_gridDefinitionDescription': 'Latitude/Longitude Grid', 'GRIB_iDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_iScansNegatively': np.int64(0), 'GRIB_jDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_jPointsAreConsecutive': np.int64(0), 'GRIB_jScansPositively': np.int64(0), 'GRIB_latitudeOfFirstGridPointInDegrees': np.float64(72.0), 'GRIB_latitudeOfLastGridPointInDegrees': np.float64(30.0), 'GRIB_longitudeOfFirstGridPointInDegrees': np.float64(-15.0), 'GRIB_longitudeOfLastGridPointInDegrees': np.float64(41.0), 'GRIB_missingValue': np.float64(3.4028234663852886e+38), 'GRIB_name': 'V component of wind', 'GRIB_shortName': 'v', 'GRIB_totalNumber': np.int64(0), 'GRIB_units': 'm s**-1', 'long_name': 'V component of wind', 'units': 'm s**-1', 'standard_name': 'northward_wind'}

  - w
    Dimensions: ('valid_time', 'pressure_level', 'latitude', 'longitude')
    Shape: (24, 2, 169, 225)
    Attributes: {'GRIB_paramId': np.int64(135), 'GRIB_dataType': 'an', 'GRIB_numberOfPoints': np.int64(38025), 'GRIB_typeOfLevel': 'isobaricInhPa', 'GRIB_stepUnits': np.int64(1), 'GRIB_stepType': 'instant', 'GRIB_gridType': 'regular_ll', 'GRIB_uvRelativeToGrid': np.int64(0), 'GRIB_NV': np.int64(0), 'GRIB_Nx': np.int64(225), 'GRIB_Ny': np.int64(169), 'GRIB_cfName': 'lagrangian_tendency_of_air_pressure', 'GRIB_cfVarName': 'w', 'GRIB_gridDefinitionDescription': 'Latitude/Longitude Grid', 'GRIB_iDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_iScansNegatively': np.int64(0), 'GRIB_jDirectionIncrementInDegrees': np.float64(0.25), 'GRIB_jPointsAreConsecutive': np.int64(0), 'GRIB_jScansPositively': np.int64(0), 'GRIB_latitudeOfFirstGridPointInDegrees': np.float64(72.0), 'GRIB_latitudeOfLastGridPointInDegrees': np.float64(30.0), 'GRIB_longitudeOfFirstGridPointInDegrees': np.float64(-15.0), 'GRIB_longitudeOfLastGridPointInDegrees': np.float64(41.0), 'GRIB_missingValue': np.float64(3.4028234663852886e+38), 'GRIB_name': 'Vertical velocity', 'GRIB_shortName': 'w', 'GRIB_totalNumber': np.int64(0), 'GRIB_units': 'Pa s**-1', 'long_name': 'Vertical velocity', 'units': 'Pa s**-1', 'standard_name': 'lagrangian_tendency_of_air_pressure'}

Global Attributes:
  GRIB_centre: ecmf
  GRIB_centreDescription: European Centre for Medium-Range Weather Forecasts
  GRIB_subCentre: 0
  Conventions: CF-1.7
  institution: European Centre for Medium-Range Weather Forecasts
  history: 2025-05-08T10:46 GRIB to CDM+CF via cfgrib-0.9.15.0/ecCodes-2.39.0 with {"source": "tmpj57q_jt0/data.grib", "filter_by_keys": {"stream": ["oper"], "stepType": ["instant"]}, "encode_cf": ["parameter", "time", "geography", "vertical"]}
