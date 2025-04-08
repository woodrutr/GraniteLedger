#####
### Set-up: Loading packages (included in Conda environment)
#####
libraries <- c("chron","data.table", "lmtest", "fBasics", "readxl", 
               "lubridate","caret")
invisible(lapply(libraries, library, character.only = TRUE))

# snakemake messes up timezone maps, need to manually reset
dir = lubridate:::tzdir_find()
print(Sys.getenv("TZDIR"))
Sys.setenv(TZDIR = dir)

### Ensure working directory pointed at Snakefile location
here::i_am("Snakefile")

# filepaths
#base_dir = 'N:/NextGen Developers/Projects/demand_profiles/BA-NOAA'
base_dir = 'outputs/BA-NOAA/'
write_to_dir = 'outputs/load-predict/'
input_map_dir = 'inputs/maps/'

# imports
tz_map = read.csv(paste0(input_map_dir,'tz_map.csv'))
source('scripts/weather_functions.R')

# settings: new fit and new prediction?
fit_data = TRUE
pred_data = TRUE

# name of new directory for fit
lm_fit_name = 'final'
# create necessary output directories
save_loc = paste0('outputs/tmp_files/',lm_fit_name,'_fit/')

dir.create('outputs/tmp_files', showWarnings = FALSE)
dir.create(save_loc, showWarnings = FALSE)
dir.create(write_to_dir, showWarnings = FALSE)

##################################

# read in demand data and clean
dem3 = read_demand()
dem3 = clean_demand(dem3,input_map_dir)

# pull all BA region names
all_bas = list.files(path=base_dir)
region_names = unique(gsub('BA-NOAA_','',gsub('.{9}$','',all_bas)))
region_names = region_names[!grepl('Snakemake|Merged',region_names)]


#####
# run demand fit and prediction
#####
# 
# The weather data used for the regression consisted of:
# the heating degree hour*, 
# the cooling degree hour*
# (both of these were interacted with a dummy variable specifying whether it was a weekday or a weekend day), 
# the previous day's heating degree day*, 
# the previous day's cooling degree day*,
# a three-hour rolling average relative humidity,
# cold wind speed*,
# and hot wind speed*.
# *The heating/cooling degree cutoff (also determining hot vs cold wind) is 65 degrees F.
#
# Other non-weather data used for the regression:
# day of week
# hour of day
# year

dem_reg_all = data.table()
for (region_name in region_names){
  print(region_name)
  all_files_reg = list.files(path = base_dir, pattern=paste0(region_name,'_'))
  
  # if not fitting or predicting, pull from already intermediate fit files
  if (!fit_data & !pred_data){
    weektype = 1
    tmp_load = data.table()
    for (month in seq(1,12)){
        file_save_csv = paste0(write_to_dir,region_name,'_',month,'_',weektype,'_predict.csv')
        
        if (file.exists(file_save_csv)){
          tmp_load1 = data.table(read.csv(file_save_csv))
          tmp_load = rbind(tmp_load,tmp_load1)
          
          rm(tmp_load1)
          gc()
        } else{
          print('no prediction data to pull')
        }
    }
  } else{ #otherwise, run everything
    load = read_weather_reg(base_dir,all_files_reg, region_name)
    tmp = run_1_reg(base_dir,region_name,weektype, month, load, dem3, all_files_reg,lm_fit_name, fit_data,pred_data)
    if (length(tmp) > 0){
      tmp_load = tmp
    }
  }
  if (exists('tmp_load')){
    # aggregate into new regions
    dem_reg = agg_data_reg(region_name,tmp_load,dem3)
    
    rm(tmp_load)
    gc()
    dem_reg_all = rbind(dem_reg_all,dem_reg)
    rm(dem_reg)
  }
  gc()
}
rm(dem3)
gc()

# drop real 2020 data due to load abnormalities, use fitted load data instead
dem_reg_all = dem_reg_all[(data_type == 'real' & Year != 2020) | (data_type == 'fitted' & (Year < 2019 | Year == 2020))]
dem_reg_all = unique(dem_reg_all)

# reassign regions
dem_reg_all1 = real_reassign_ba(copy(dem_reg_all))
dem_reg_all1$Sub.Region = NULL
gc()

# aggregate years
dem_reg_all1 = year_aggregation(dem_reg_all1)
gc()

# manual renaming of PNM to PNM_PNM
dem_reg_all1[PCA_SHORT_NAME=='PNM',PCA_SHORT_NAME:=paste0(PCA_SHORT_NAME,'_PNM')]

# write out aggregated regression load to write_to_dir
write.csv(dem_reg_all1,
          paste0(write_to_dir,'LoadShapesData_predict_monthly_ol.csv'),
          row.names=FALSE)

# delete Rplots.pdf if it exists
f <- "Rplots.pdf"
if (file.exists(f)) {
  file.remove(f)
}
