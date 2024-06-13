import climate_learn as cl

root_directory = "Project/climate-learn_dataset"
variables = ["constants_5.625deg", "toa_incident_solar_radiation_5.625deg", "2m_temperature_5.625deg", "10m_u_component_of_wind_5.625deg",
             "10m_v_component_of_wind_5.625deg", "geopotential_5.625deg", "u_component_of_wind_5.625deg", "v_component_of_wind_5.625deg",
             "temperature_5.625deg", "specific_humidity_5.625deg", "relative_humidity_5.625deg"]
names = ["constants", "toa_incident_solar_radiation", "2m_temperature", "10m_u_component_of_wind",
             "10m_v_component_of_wind", "geopotential", "u_component_of_wind", "v_component_of_wind",
             "temperature", "specific_humidity", "relative_humidity"]

for i in range(len(variables)):
  cl.data.download_weatherbench(
      dst=f"{root_directory}/{variables(i)}",
      dataset="era5",
      variable=names(i),
      resolution=5.625)
