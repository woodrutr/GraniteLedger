days = c('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')

# parameters used for relative humidity calculation
beta = 17.625
lambda = 243.04

#add crosswalk of actual regions to map-made regions and aggregate
new_regionality = function(df,cw,merge_cols,old_col,new_col){
  cw = data.table(read.csv(cw))
  cw[is.na(get(new_col)),(new_col):='NA']
  df = merge(df,cw,
             by=merge_cols,
             all.x=TRUE)
  df[is.na(get(new_col)),(new_col):=get(old_col)]
  df[,old_col] = NULL
  setnames(df,new_col,old_col)
  return(df)
}

# preprocess weather data to create variables for fitting
preprocess_data = function(load2){
  # remove 10 weightings
  load2[,dp_real:=tmp_dew_celsius_10_weighted/10]
  load2[,temp_real:=tmp_air_celsius_10_weighted/10]
  
  # heating/cooling degree cutoff temp
  temp_cut = 65
  
  # convert C to F
  load2[,temp_f:=temp_real*9/5 + 32]
  
  # heating/cooling degree hour determination
  load2[,hdh:=0]
  load2[,cdh:=0]
  load2[temp_f<temp_cut,hdh:=abs(temp_f- temp_cut)]
  load2[temp_f>temp_cut,cdh:=abs(temp_f- temp_cut)]
  
  # calculate previous day's average temp
  load2[,temp_f_avg:=mean(temp_f),by=actual_date]
  load2[,temp_f_avg_prev:=shift(temp_f_avg,24)]
  load2[is.na(temp_f_avg_prev),temp_f_avg_prev:=temp_f_avg]
  
  # previous day's cooling/heating degree days
  load2[,hdd_prev:=0]
  load2[,cdd_prev:=0]
  load2[temp_f_avg_prev<temp_cut,hdd_prev := abs(temp_f_avg_prev- temp_cut)]
  load2[temp_f_avg_prev>temp_cut,cdd_prev := abs(temp_f_avg_prev- temp_cut)]
  
  # hot and cold wind speed
  load2[,wind_hot:=0]
  load2[,wind_cold:=0]
  load2[cdh>0,wind_hot:=wind_spd_m_per_sec_10_weighted]
  load2[hdh>0,wind_cold:=wind_spd_m_per_sec_10_weighted]
  
  # rolling 3-hour relative humidity calculation
  load2[,rel_humidity:=100*(exp((beta*dp_real)/(lambda+dp_real)))/(exp((beta*temp_real)/(lambda+temp_real)))]
  load2[,paste0('hum_3',seq(1,2)):= shift(rel_humidity,seq(1,2))]
  load2[,hum_3:=rel_humidity]
  load2[!is.na(hum_31),hum_3:=(hum_3+hum_31)/2]
  load2[!is.na(hum_32),hum_3:=(hum_3*2+hum_32)/3]
  
  if (!('local_time' %in% colnames(load2))){
    load2[,local_time2:=paste0(actual_date,sprintf(" %02d",Hour),':00:00')]
  }
  if (!('Hour' %in% colnames(load2))){
    load2[,Hour:=format()]
  }
  
  # assign hour of day param
  load2[,paste0('hr',seq(1,24)) := 0]
  for (h in seq(1,24)){
    hr_name = paste0('hr',h)
    load2[Hour == h,c(hr_name) := 1]
  }
  
  load2$weekday = weekdays(load2$actual_date) #change this to local time
  load2[,weekend:=0]
  load2[is.holiday(actual_date),weekend:=1]
  load2[weekday %in% c('Saturday','Sunday'),weekend:=1] 
  
  # separate heating/cooling degree hours by weekday/weekend
  load2[,hdh_end:=0]
  load2[,cdh_end:=0]
  load2[,hdh_day:=0]
  load2[,cdh_day:=0]
  load2[weekend == 1,hdh_end:=hdh]
  load2[weekend == 1,cdh_end:=cdh]
  load2[weekend == 0,hdh_day:=hdh]
  load2[weekend == 0,cdh_day:=cdh]
  
  # assign day of week (wd_day) (holiday = Sunday)
  load2[,paste0('wd_',days) := 0]
  load2[(weekend & !(weekday %in% c('Saturday','Sunday'))),weekday := 'Sunday']
  for (w in days){
    wd_name = paste0('wd_',w)
    load2[weekday == w,c(wd_name) := 1]
  }
  
  # create year params
  load2[,Year2019:=0]
  load2[,Year2022:=0]
  load2[,Year2021:=0]
  load2[Year== 2019,Year2019:=1]
  load2[Year== 2022,Year2022:=1]
  load2[Year== 2021,Year2021:=1]
  
  return(load2)
}

# reassigns/aggregates regions
real_reassign_ba = function(df){
  df[,c('Balancing.Authority_real','Sub.Region'):=tstrsplit(Balancing.Authority, "-", fixed=TRUE)] 
  df[,Balancing.Authority:=NULL]
  setnames(df,'Balancing.Authority_real','Balancing.Authority')
  
  df = df[,.(Demand..MW.=mean(Demand..MW.)),
          by=.(Balancing.Authority,Sub.Region,Date,Hour,Month,Year,weekend,data_type)]
  
  df = new_regionality(df,
                       'inputs/maps/ba_new.csv',
                       c('Balancing.Authority'),
                       'Balancing.Authority',
                       'BA_New')
  
  df = new_regionality(df,
                       cw = 'inputs/maps/basr_new.csv',
                       merge_cols = c('Balancing.Authority','Sub.Region'),
                       old_col='Sub.Region',
                       new_col='BASR_New')
  
  df[!is.na(Sub.Region) & Sub.Region != 'NA',
     Balancing.Authority:=paste0(Balancing.Authority,'_',Sub.Region)]
  return(df)
  
}

# Read in 930 load data
read_demand = function(){
  
  dem = data.table(read.csv('outputs/LoadCleaned.csv'))
  
  dem = dem[,.(Demand..MW.=sum(Demand..MW.)),by=.(Balancing.Authority,Date,Hour,Month,UTC.Time.at.End.of.Hour)]
  dem[,max_dem:=max(Demand..MW.),by=.(Balancing.Authority,Date, Hour, Month)]
  
  dem_local = dem[Demand..MW. == max_dem][,c('Balancing.Authority',       'Date' , 'Hour' ,'Month', 'UTC.Time.at.End.of.Hour')]
  #check no duplicate UTC per region
  
  dem2 = dem[,.(Demand..MW.=sum(Demand..MW.)),by=.(Balancing.Authority,Date, Hour, Month)]
  dem3 = merge(dem2,dem_local,by=c('Balancing.Authority','Date', 'Hour', 'Month'))
  
  suppressWarnings(dem3[,actual_date:=as.Date(Date)])
  dem3$weekday = weekdays(dem3$actual_date) #change this to local time
  dem3[,weekend:=0]
  
  dem3[is.holiday(actual_date),weekend:=1]
  dem3[weekday %in% c('Saturday','Sunday'),weekend:=1] 
  dem3[, Year := tstrsplit(Date, "-", fixed=TRUE)[1]]
  
  #exclude ERCOT data
  dem3 = dem3[!(grepl('ERCO',Balancing.Authority) & Year < 2020 & Month < 6)]
  
  return(dem3)
}

# clean 930 load data
# assigns new regions, changes regional naming conventions, remove bad data
clean_demand = function(dem3, input_map_dir){
  dem3[!grepl('_',Balancing.Authority),Balancing.Authority:=paste0(Balancing.Authority,'_NA')]
  dem3[,c('Balancing.Authority_real','Sub.Region'):=tstrsplit(Balancing.Authority, "_", fixed=TRUE)] #check the - vs _
  dem3[,Balancing.Authority:=NULL]
  setnames(dem3,'Balancing.Authority_real','Balancing.Authority')
  
  dem3 = new_regionality(df=dem3,
                         cw = paste0(input_map_dir,'basr_new_noaa.csv'),
                         merge_cols = c('Balancing.Authority','Sub.Region'),
                         old_col='Sub.Region',
                         new_col='BASR_New')
  
  dem3[,Balancing.Authority:=paste0(Balancing.Authority,'-',Sub.Region)]
  dem3[!grepl('-',Balancing.Authority),Balancing.Authority:=paste0(Balancing.Authority,'-NA')]
  unique(dem3$Balancing.Authority)
  dem3$Sub.Region = NULL
  
  dem3 = dem3[,.(Demand..MW.=sum(Demand..MW.)),
              by=.(Balancing.Authority, Date, Hour, Month, Year, weekend,actual_date, UTC.Time.at.End.of.Hour)]
  dem3[,dayofmonth:=as.numeric(substr(Date, nchar(Date)-1, nchar(Date)))]
  
  #removing ERCOT energy crisis data
  dem3 = dem3[!(grepl('ERCO',Balancing.Authority) & Year == 2021 & Month == 2 & dayofmonth %in% seq(10,27))]
  
  return(dem3)
}

# Read in NOAA weather data
read_weather_reg = function(base_dir,all_files_reg, reg){
  load_all = data.table()
  for (f in all_files_reg){
    load = data.table(read.csv(paste0(base_dir,'/',f)))
    load[,Balancing.Authority := reg]
    load[,utc_time:=paste0(sprintf("%04d-%02d-%02d %02d", year,month, day,hour),':00:00')]
    load_all = rbind(load_all, load)
  }
  
  return(load_all)
}

# creates model fit
# writes fit out to RDS file, writes out fit statistics
fit_model_reg = function(load2,lm_fit_name, region_name, weektype,m){
  
  # specifications of how to model,
  # coming from somewhere else
  outcome <- "Demand..MW."
  
  # choose variables to fit with
  variables <- c("hum_3",
                 'hdh_end','hdh_day','cdh_end','cdh_day',
                 'hdd_prev','cdd_prev',
                 'wind_hot','wind_cold',
                 paste0('hr',seq(1,23)),
                 paste0('wd_',days[-1])
  )
  if (nrow(load2[Year2019 == 0]) > 3){ #non 2019 values
    variables = c(variables,"Year2019")
    
    if (nrow(load2[Year2022 >0]) > 3){
      variables = c(variables,"Year2022")
    }
  }
  
  # our modeling effort, 
  # fully parameterized!
  f <- as.formula(
    paste(outcome, 
          paste(variables, collapse = " + "), 
          sep = " ~ "))
  
  model <- lm(f, data = load2)
  
  summary(model)
  coefficients(model)
  
  results_value <- data.frame(region = region_name,
                              Month = m,
                              weekend=weektype,
                              R2 = summary(model)$r.squared
                              #                            RMSE = RMSE(model, dem_load),
                              #                            MAE = MAE(model, dem_load)
  )
  
  #save model to Rdata so we can predict later
  save_loc = paste0('outputs/tmp_files/',lm_fit_name,'_fit/')
  
  file_save = paste0(save_loc,region_name,'_',m,'_',weektype,"_model.RDS")
  saveRDS(model, file = file_save) 
  
  
  tmp = summary(model)$coefficients[, "Pr(>|t|)"]
  prob_tmp = transpose(as.data.table(tmp))
  setnames(prob_tmp,names(prob_tmp),names(tmp))
  prob_tmp[,reg:=region_name]
  
  load2[,lm_fitted:=fitted.values(model)]
  load2[,lm_residuals:=residuals(model)]
  load2[,diff:=lm_fitted-Demand..MW.]
  
  load2[,wd_plot := 0]
  load2[weekday == 'Monday',wd_plot:=1]
  load2[weekday == 'Tuesday',wd_plot:=2]
  load2[weekday == 'Wednesday',wd_plot:=3]
  load2[weekday == 'Thursday',wd_plot:=4]
  load2[weekday == 'Friday',wd_plot:=5]
  load2[weekday == 'Saturday',wd_plot:=6]
  load2[weekday == 'Sunday',wd_plot:=7]
  
  coeff_list = coefficients(model)
  coeff_tmp = as.data.table(coeff_list,keep.rownames = TRUE)
  coeff_tmp = transpose(coeff_tmp, keep.names = "col", make.names = "rn")
  coeff_tmp$col = NULL
  coeff_tmp[,reg:=region_name]
  
  # open file to write plots
  png(filename=paste0('outputs/tmp_files/',lm_fit_name,'_fit/',region_name,'_',m,'_',weektype,'_residuals.png'))
  
  # create matrix of plots
  layout(matrix(c(1,2,3,4,5,6,7,8,9),3,3,byrow=T))
  
  # LOESS scatter plot of residuals
  plot(model, which = 1, col = "blue")
  
  #acf plot
  #acf(model$residuals)
  
  # Q-Q plot of linear model
  plot(model, which = 2, col = "blue")
  
  # scatter plot of standardized residuals
  plot(model, which = 3, col = "blue")
  
  plot(model, which = 5, col = "blue")
  
  plot(cooks.distance(model), pch = 16, col = "blue")
  
  plot(load2$Hour, model$resid,
       main="Residuals by Hour",
       xlab="Hour",ylab="Residuals")
  abline(h=0,lty=2)
  
  plot(load2$wd_plot, model$resid,
       main="Residuals by weekday",
       xlab="Weekday",ylab="Residuals")
  abline(h=0,lty=2)
  
  plot(load2$temp_real, model$resid,
       main="Residuals by temp_real",
       xlab="temp_real",ylab="Residuals")
  abline(h=0,lty=2)
  
  plot(load2$Year, model$resid,
       main="Residuals by Year",
       xlab="Year",ylab="Residuals")
  abline(h=0,lty=2)
  
  dev.off() # close plot file
  
  #Testing normal distribution and independence assumptions
  print(jarqueberaTest(model$resid)) #Test residuals for normality
  
  plot(cooks.distance(model), pch = 16, col = "blue")
  
  #test if residual variances are equal (we want it to be): if p-value > 0.05
  print(bptest(model))
  
  return(model)
}

# process prediction data
process_pred_data = function(tmp_load,region_name){
  ba_reg = unique(strsplit(region_name,'-'))[[1]][1]
  print(ba_reg)
  tmp_load[,ba:=ba_reg]
  tmp_load = merge(tmp_load,tz_map, by='ba')
  tmp_load[,tz2:=paste0('US/',tz)]
  tz2 = tmp_load$tz2[1]
  print(tz2)
  tmp_load$local_time = as.POSIXct(tmp_load$utc_time,
                                   format='%Y-%m-%d %H:%M:%S',
                                   tz='UTC') 
  tmp_load[,local_time := local_time-3600] #subtract 1 hour of time and correct later...
  

  tmp_load$local_time2 = copy(lubridate::with_tz(tmp_load$local_time, tzone = tz2)) 
  tmp_load$local_time2 = format(copy(tmp_load$local_time2), 
                                format='%Y-%m-%d %H:%M:%S' )

  
  tmp_load$local_time = as.POSIXct(tmp_load$local_time2,
                                   format='%Y-%m-%d %H:%M:%S',
                                   tz='UTC') 
  
  tmp_load$Hour =  as.numeric(format(tmp_load$local_time, '%H' ))
  tmp_load$Year =  as.numeric(format(tmp_load$local_time, '%Y' ))
  tmp_load$Month =  as.numeric(format(tmp_load$local_time, '%m' ))
  tmp_load$Day =  as.numeric(format(tmp_load$local_time, '%d' ))
  
  #re-add 1 hour to Hour for formatting
  tmp_load[,Hour:=Hour+1]
  
  tmp_load$actual_date = as.Date(tmp_load$local_time2)
  tmp_load = preprocess_data(tmp_load)
  tmp_load[,c('hour','year','month','day','local_time'):=NULL]
  
  return(tmp_load)
}

# predict load data for a given region/month based on fit
predict_model_reg = function(tmp_load,lm_fit_name, region_name,weektype,m){
  #save model to RDS so we can predict later
  save_loc = paste0('outputs/tmp_files/',lm_fit_name,'_fit/')
  file_save = paste0(save_loc,region_name,'_',m,'_',weektype,"_model.RDS")
  
  if (!file.exists(file_save)){
    print('WARNING: NOAA region has no 930 load match')
  } else{
    model_saved <- readRDS(file_save)
    pred_results = predict(model_saved, newdata=tmp_load) #,se.fit = TRUE)
    tmp_load$predicted_demand = pred_results #$fit
    
    file_save_csv = paste0(write_to_dir,region_name,'_',m,'_',weektype,"_predict.csv")
    
    tmp_load  = tmp_load[,c('ba','Balancing.Authority','Year',	'Month',	'Day',	'Hour',	'actual_date',
                            'tmp_air_celsius_10_weighted',	'tmp_dew_celsius_10_weighted',	'wind_dir_degrees_weighted',	'wind_spd_m_per_sec_10_weighted',
                            'temp_real','rel_humidity',
                            'weekend','predicted_demand')] 
    
    write.csv(tmp_load, file_save_csv,row.names=FALSE)
    
    # remove because too large of files
    if (file.exists(file_save)) {
      #Delete file if it exists
      file.remove(file_save)
    }
    
    return(tmp_load)
  }
}

# fit and predict load data for a given region
run_1_reg = function(base_dir,region_name, weektype, month, load, dem3,all_files_reg,lm_fit_name,fit_data, pred_data){
  
  # process data fitting
  if (fit_data){
    load2 = merge(load,
                  dem3,
                  by.y=c('UTC.Time.at.End.of.Hour', 'Balancing.Authority'),
                  by.x=c('utc_time','Balancing.Authority'))
    if (nrow(load2) == 0){
      print('WARNING: NOAA region has no 930 load match')
    } else{
      
      load2 = preprocess_data(load2)
      load2[, Year := tstrsplit(Date, "-", fixed=TRUE)[1]]
      
      load2 = load2[Year != 2020]
      load2[,dayofmonth:=as.numeric(substr(Date, nchar(Date)-1, nchar(Date)))]
      
      # dropping ercot power crisis data
      load2 = load2[!(grepl('ERCO',Balancing.Authority) & Year == 2021 & Month == 2 & dayofmonth %in% seq(10,27))]
      
      #m = month
      for (m in seq(1,12)){
        load3 = copy(load2)[Month == m] 
        
        model = fit_model_reg(load3,lm_fit_name, region_name, weektype=1,m)
        load3$predict_load = model$fitted.values
        load3$cooks.distance= cooks.distance(model)
        
        print(summary(model))
      }
    }
  }
  
  # process prediction step
  if (pred_data){
    tmp_load_all = data.table()
    tmp_load_save = process_pred_data(load,region_name)
    weektype=1
    for (m in seq(1,12)){
      save_loc = paste0('outputs/tmp_files/',lm_fit_name,'_fit/')
      file_save = paste0(save_loc,region_name,'_',m,'_',weektype,"_model.RDS")
      if (!file.exists(file_save)){
        print('WARNING: NOAA region has no fitted model')
      } else{
        tmp_load = tmp_load_save[Month == m & weekend == weektype]
        
        tmp_load = predict_model_reg(tmp_load,lm_fit_name, region_name,weektype, m)
        tmp_load_all = rbind(tmp_load_all,tmp_load)
      }
    }
  }
  
  if (exists('tmp_load_all')){
    return_list = tmp_load_all
  } else{
    return_list = data.table()
  }
  return(return_list)
}

# for a given region, aggregate real and fitted data
agg_data_reg = function(region_name,tmp_load,dem3){
  # get real load (2019-2022)
  real_load = dem3[Balancing.Authority == region_name]
  real_load = real_load[,c('Balancing.Authority','Date','Hour','Month','Year','Demand..MW.','weekend')]
  
  # get fitted load
  fitted_load = tmp_load[,c('Balancing.Authority','actual_date','Hour','Month','Year','predicted_demand','weekend')]
  setnames(fitted_load,
           c('actual_date','predicted_demand'),
           c('Date','Demand..MW.'))
  fitted_load[,Date:=as.character(Date)]
  fitted_load[,Year:=as.character(Year)]
  fitted_load$data_type = 'fitted'
  
  real_load$data_type = 'real'
  
  df = rbind(fitted_load, real_load) 
  return(df)
}

# average all years' demand
year_aggregation = function(df){
  df[,c('Year', 'Month','Day') := tstrsplit(Date,'-',fixed=TRUE)]
  df[,Year:=as.integer(Year)]
  df[,Month:=as.integer(Month)]
  df[,Day:=as.integer(Day)]
  
  # sum demand over new regions
  df = df[,.(Demand..MW.=sum(Demand..MW.)), by=.(Balancing.Authority,Month,Day,Hour,Year)]
  
  # leap days: agg with 2/28
  df[Month == 2 & Day == 29 , Day:=28]
  
  # find mean demand over all years
  df = df[,.(Demand..MW.=mean(Demand..MW.)), by=.(Balancing.Authority,Month,Day,Hour)]
  df = df[order(Balancing.Authority,Month,Day,Hour)]
  
  # final formatting
  setnames(df,c('Balancing.Authority','Demand..MW.'),
           c('PCA_SHORT_NAME','Value'))
  return(df)
}