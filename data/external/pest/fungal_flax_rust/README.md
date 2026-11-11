# A multi-year case study highlighting the influence of hydrological conditions on epidemic dynamics in a natural plant pathosystem

Please reach out to Keenan Duggal ([keenand@princeton.edu](mailto:keenand@princeton.edu)) with any questions about the data and/or scripts associated with this manuscript:

[https://doi.org/10.5061/dryad.98sf7m0tc](https://doi.org/10.5061/dryad.98sf7m0tc)

## Description of the data and file structure

This ecological study was purely observational, and as such no experimental efforts were taken during the collection of data.

### Overview:

There are four field sites where observations of flax rust epidemiology were made.  They are first described in 

`Miller, I.F., Jiranek, J., Brownell, M. et al. Predicting the effects of climate change on the cross-scale epidemiological dynamics of a fungal plant pathogen. Sci Rep 12, 14823 (2022). https://doi.org/10.1038/s41598-022-18851-z` as follows: 

* "The lowest site, ‘cement creek’ (CC; approximately 38.82156° N, 106.86893° W), falls within a sage brush meadow on the boundary between the mountain shrub and montane vegetation zones at 2440 m elevation. The second lowest site, ‘bus turnaround’ (BT; approximately 38.97130° N, 106.99595° W), is situated in an open subalpine meadow at the base of Gothic Mountain at ~ 2940 m. The second highest site, ‘gothic mountain’ (GM; approximately 38.97969° N, 107.01937° W), is also in the subalpine zone on a steep hillside within a clearing in an evergreen forest on the lower slopes of Gothic Mountain at 3220 m. The highest site, ‘high meadow’ (HM; approximately 38.96779° N, 107.02184° W), is on an exposed meadow at the upper fringes of the subalpine zone on the shoulder of Gothic Mountain at 3,410 m."

**Key   of Common Variables in this Dataset**:

*Site (abbreviations):* 

* BT: Bus Turnaround 
* CC: Cement Creek 
* GM: Gothic Mountain 
* HM: High Meadow

*Each site was gridded  into a cartesian coordinate system:*

* X: X coordinate of plant (between 0 - 9 meters) in gridded survey plot 
* Y: Y coordinate of plant (between 0 - 19 meters) in gridded survey plot 
* x: Add to X coordinate for precise location within gridded survey plot (between 0 - 1 meters) 
* y: Add to Y coordinate for precise location within gridded survey plot (between 0 - 1 meters) 

Example: If the location of a plant is described by X=2, x=0.5, Y=4, y=0.9, it can be found 2.5 meters from the left border and 4.9 meters from the bottom border of the site. 

*Other Common Variables:* 

* Year: Year of observation
* Disease class (Class(0-5)): 
  * 0: 0% of plant tissue is visibly infected 
  * 1: 0-20 % of plant tissue is visibly infected 
  * 2: 20-40 % of plant tissue is visibly infected 
  * 3: 40-60 % of plant tissue is visibly infected 
  * 4: 60-80 % of plant tissue is visibly infected 
  * 5: 80-100 % of plant tissue is visibly infected 
* Tag: Unique numerical identifier for a  plant (~10% of plants were tagged in each site)
* Date (or Date Observed): Date that survey was performed
* Notes: Not used for analyses but rather for internal record keeping
* days*_*since_june_1st: # days of since June 1st of the current year 

\*** Important note: Blank cells in any of the following files do not imply missing data. Some of the reasons for empty cells:*

* * In different weeks and on different days, different disease measurements were taken (e.g., within-host infection quantification vs between-host epidemiological dynamics tracking), and these data were aggregated into files like "Epidemiology."  *
* *In different years, different data was collected (e.g., 2023 was the first year that used the streamlined "Disease Class" measurement to approximate within-host intensity. *
* *Sometimes, "Notes" were made to describe findings from the field. *
* *Not all plants were "tagged", but those that were have additional measurements and data associated with them *

*Please contact the corresponding author if there is any confusion of how to use this data. *

### Files and variables

#### File: 2023\_Epidemiology\_Final\_Total\_Surveys.xlsx

**Description:** End-of-the-year disease surveys for each site. 

##### Unique Variables

* New?: 1 indicates that the plant was found newly diseased on the date of observation. Blank indicates it is not newly diseased. 

#### File: cleaned\_daily\_temp.rds

**Description:** Temperature (degrees Celsius), relative humidity (%) and dew point (degrees Celsius) measurements were collected continuously every five minutes for the duration of the observational period. For this file, the daily average of each variable in each site and year was calculated and the mean, maximum and minimum* *of each of these is reported below. Temp: temperature, rh: relative humidity, dp: dew point.

##### Unique Variables

* mean_temp: 
* min_temp
* max_temp
* mean_rh
* min_rh
* max_rh
* mean_dp
* min_dp
* max_dp

#### File: cleaned\_daily\_weather.rds

**Description:** Leaf moisture (leaf_wetness), rainfall (rain), sunlight intensity (solar_radiation), wind speed (ws), wind direction (wd), gust speed (gust_speed), soil moisture (water_content) measurements were collected continuously every five minutes for the duration of the observational period. For this file, the daily average of each variable in each site and year was calculated and the mean, maximum and minimum* *of each of these is reported below. For instrumentation details and descriptions of units, please refer to 

`Miller, I.F., Jiranek, J., Brownell, M. et al. Predicting the effects of climate change on the cross-scale epidemiological dynamics of a fungal plant pathogen. Sci Rep 12, 14823 (2022). https://doi.org/10.1038/s41598-022-18851-z`

##### Unique Variables

* "mean_leaf_wetness"   "min_leaf_wetness"     "max_leaf_wetness"     "mean_rain"            "min_rain" "max_rain"             "mean_solar_radiation" "min_solar_radiation"  "max_solar_radiation" "mean_ws"              "min_ws"               "max_ws"               "mean_wd"              "min_wd"               "max_wd"               "mean_gust_speed"      "min_gust_speed"       "max_gust_speed"       "mean_water_content"   "min_water_content"    "max_water_content"

#### File: Cleaned\_Epi.rds

**Description:** Summarized epidemiological data used for analyses 

##### Unique Variables

* Disease_date: Date of survey 
* count: total number of plants in site
* new_infections: # newly infected plants observed
* prevalence_noRecov: prevalence (total infections / count) without factoring in recovery 
* prevalence: prevalence (total infections / count) with factoring in recovery 
* Date_yearless: Date with common year (2000) -- included to have a common axis in analyses

#### File: Population\_density.xlsx

**Description:** Mapping the total density and approximate locations of all plants in each site 

##### Unique Variables

* \#H: Number of healthy plants in each 1mX1m grid
* \#D: Number of diseased plants in each 1mX1m grid
* \#Seedlings: Number of seedlings  in each 1mX1m grid

#### File: Plant\_locations.xlsx

**Description:**  Mapping the total density and approximate locations of all plants in each site 

##### Unique Variables

* Status: Healthy (H), Diseased (D), X (Dead), L (Lost) 
* Height.cm: Maximum height of plant in centimeters 

#### File: Epidemiology.xlsx

**Description:** Raw epidemiological survey data 

##### Unique Variables

* Date First Observed Diseased: Date that infection was first spotted on plant 
* Total # Stems: Total number of stems on plant 
* Total # Diseased Stems: Total number of diseased stems on plant 
* Max Height: Maximum height of plant in cm 
* Flowers: Yes or No indicating whether plant was flowering at time of survey 

#### File: cleaned\_weather.rds

**Description:**  Leaf moisture (leaf_wetness), rainfall (rain), sunlight intensity (solar_radiation), wind speed (ws), wind direction (wd), gust speed (gust_speed), soil moisture (water_content) measurements were collected continuously every five minutes for the duration of the observational period. For instrumentation details and descriptions of units, please refer to 

`Miller, I.F., Jiranek, J., Brownell, M. et al. Predicting the effects of climate change on the cross-scale epidemiological dynamics of a fungal plant pathogen. Sci Rep 12, 14823 (2022). https://doi.org/10.1038/s41598-022-18851-z`

##### Unique Variables

* times: Date and time

#### File: cleaned\_temp.rds

**Description:** Temperature (degrees Celsius), relative humidity (%) and dew point (degrees Celsius) measurements were collected continuously every five minutes for the duration of the observational period. Temp: temperature, RH: relative humidity, Dew_point: dew point.

##### Unique Variables

* times: Date and time

## Code/software

*All code was run using R version 4.2.3. Packages needed are included at the top of each script.* 

**Analyses_and_Figures.R**  : is a script that includes all of the code used for analyses and figure generation associated with this manuscript.

**epi_data_aggregation.R** : is a script that was used to clean, process and aggregate all of the epidemiological survey data from each year.

**weather_data_aggregation.R** : is a script that was used to clean, process and aggregate all of the weather  data from each year.

## Access information

Data was derived from the following sources:

* The RMBL Spatial Data Platform: [https://www.rmbl.org/scientists/resources/data-catalog/](https://www.rmbl.org/scientists/resources/data-catalog/) 
* Specifically raster data describing Elevation, Growing Degree Days (and Snow-free Growing Degree Days, & Days of contiguous snow cover were accessed using the coordinates for each site (described above). 
* Please reach out to the corresponding author for more information about how to access this data. 
