library(DBI)
library(RSQLite)
library(tidyverse)
ed_visit <- 
  read_csv("data/ed_visit.csv",
           col_types = 
             cols(
               `cohort reference event-ed visits - age at ed visit` = col_double(),
               age = col_double(),
               gender = col_character(),
               `cohort reference event-ed visits - visit start date` = col_character(),
               `past charlson-cerebrovascular disease` = col_character(),
               `past charlson-charlson score` = col_double(),
               `past charlson-chronic pulmonary disease` = col_character(),
               `past charlson-congestive heart failure` = col_character(),
               `past charlson-dementia` = col_character(),
               `past charlson-diabetes mellitus` = col_character(),
               `past charlson-hemiplegia or paraplegia` = col_character(),
               `past charlson-liver disease` = col_character(),
               `past charlson-myocardial infarction` = col_character(),
               `past charlson-peptic ulcer disease` = col_character(),
               `past charlson-peripheral vascular disease` = col_character(),
               `past charlson-renal disease` = col_character(),
               `past charlson-rheumatic disease connective tissue disease` = col_character(),
               `hr vs-numeric result` = col_double(),
               `saturation-numeric result` = col_double(),
               `systolic bp -numeric result` = col_double(),
               `diastolic bp-numeric result` = col_double(),
               `resp rate-numeric result` = col_double(),
               `temp-numeric result` = col_double(),
               `cohort reference event-ed visits - was admitted` = col_character()
             ))


colnames(ed_visit) <-
  c("age_at_ed_visit",
    "age",
    "gender",
    "start_date_time",
    "charlson_cerebrovascular_disease",
    "charlson_score",
    "charlson_chronic_pulmonary_disease",
    "charlson_congestive_heart_failure",
    "charlson_dementia",
    "charlson_diabetes_mellitus",
    "charlson_hemiplegia_or_paraplegia",
    "charlson_liver_disease",
    "charlson_myocardial_infarction",
    "charlson_peptic_ulcer_disease",
    "charlson_peripheral_vascular_disease",
    "charlson_renal_disease",
    "charlson_rheumatic_disease",
    "heart_rate",
    "saturation",
    "systolic_bp",
    "diastolic_bp",
    "resp_rate",
    "temp",
    "admitted")


ed_visit$start_date_time <- 
  strptime(ed_visit$start_date_time, "%d/%m/%Y %H:%M")

ed_visit$admitted <- tolower(ed_visit$admitted)

ed_visit <- ed_visit %>% filter(!is.na(admitted))

ed_visit <- 
  ed_visit %>% 
  mutate(across(starts_with("charlson_"), 
                function(x) as.integer(x == "Yes"))) %>% 
  mutate(female = as.integer(gender == "female")) %>% 
  select(-gender)


missing_cols <-
  ed_visit %>% 
  mutate(id = 1:nrow(.)) %>% 
  select(id, starts_with("charlson"), "female",
         "heart_rate", "saturation", "systolic_bp",
         "diastolic_bp", "resp_rate", "temp") %>% 
  select(-charlson_score) %>% 
  pivot_longer(cols = -id) %>% 
  mutate(value = as.integer(is.na(value)),
         name = paste0(name, "_missing")) %>% 
  pivot_wider(id_cols=id) %>% 
  select(-id)

ed_visit <-
  ed_visit %>% 
  mutate(across(c(starts_with("charlson"), "female",
                  "heart_rate", "saturation", "systolic_bp",
                  "diastolic_bp", "resp_rate", "temp"),
                function(x) ifelse(is.na(x), 0, x) )) 

ed_visit <- bind_cols(ed_visit, missing_cols)

heart_rate_min = 0
heart_rate_max = 216
saturation_max = 100
temp_max = 41

bad_heart_rate <- 
  ed_visit$heart_rate > heart_rate_max |
  ed_visit$heart_rate < heart_rate_min

ed_visit$heart_rate_missing[bad_heart_rate] = 1
ed_visit$heart_rate[bad_heart_rate] = 0

bad_saturation <- 
  ed_visit$saturation > saturation_max

ed_visit$saturation_missing[bad_saturation] = 1
ed_visit$saturation[bad_saturation] = 0

bad_temp <- 
  ed_visit$temp > temp_max

ed_visit$temp_missing[bad_temp] = 1
ed_visit$temp[bad_temp] = 0

ed_visit <- 
  ed_visit %>% 
  mutate(date = as.Date(start_date_time),
         hour = as.integer(strftime(start_date_time, "%H")),
         min  = as.integer(strftime(start_date_time, "%M"))) %>% 
  select(-start_date_time) 


# 
# ggplot(ed_visit, 
#        aes(y = as.integer(admitted=='yes'), x = heart_rate)) + 
#   geom_smooth()
# 
# ggplot(ed_visit %>% filter(temp!=0), 
#        aes(y = as.integer(admitted=='yes'), x = temp)) + 
#   geom_smooth()


train_date_cutoff <- 
  as.Date(quantile(as.numeric(ed_visit$date), 0.8), 
          origin = "1970-01-01")

ed_visit_train <- 
  ed_visit %>% 
  filter(date <= train_date_cutoff) %>% 
  mutate(date = strftime(date))

ed_visit_test <- 
  ed_visit %>% 
  filter(date > train_date_cutoff) %>% 
  mutate(date = strftime(date))

mydb <- dbConnect(RSQLite::SQLite(), "data/ed_visit.db")
dbWriteTable(mydb, "ed_visit_train", ed_visit_train, overwrite = TRUE)
dbWriteTable(mydb, "ed_visit_test", ed_visit_test, overwrite = TRUE)
dbDisconnect(mydb)

