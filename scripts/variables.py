## Lists of variable names from data dfs in Brighten Dataset V1 & V2
# List of df names
df_names = ['v1_day', 'v2_day', 'v1_week', 'v2_week']
df_mis = ['v1_day_mis','v2_day_mis','v1_week_mis','v2_week_mis']
df_names_with_mis = df_names + df_mis

# Variables to keep consistent over all dfs and to not alter in preprocessing
id_columns=['num_id','dt','week','day','week_id','idx','v','season']
demographic_columns = ['gender','age','num_id']

## Variables from Baseline Surveys
gad_cols = ['gad7_1','gad7_2','gad7_3','gad7_4','gad7_5','gad7_6','gad7_7','gad7_8','gad7_sum','gad_cat']
phq9_base = ['phq9_1_base','phq9_2_base','phq9_3_base','phq9_4_base','phq9_5_base','phq9_6_base','phq9_7_base','phq9_8_base','phq9_9_base','phq9_sum_base']
mania = ['screen_1','screen_2','screen_3','screen_4']
demographics = ['gender','education','working','income_satisfaction','income_lastyear','marital_status','race','age','heard_about_us','device']
alc_cols = ['alc_1','alc_2','alc_3','alc_sum']

baseline_cols = gad_cols + phq9_base + mania + demographics + alc_cols


## Variables from Weekly Surveys
phq9_cols = ['phq9_1','phq9_2','phq9_3','phq9_4','phq9_5','phq9_6','phq9_7','phq9_8','phq9_9','phq9_sum', 'phq9_bin','phq9_cat']
sds_cols = ['sds_1','sds_2','sds_3','stress','support']
sleep_cols = ['sleep_1','sleep_2','sleep_3']
gic_cols = ['mood_1']
mhs_cols = ['mhs_1','mhs_2','mhs_3','mhs_4','mhs_5']

weekly_cols = phq9_cols + sds_cols + sleep_cols + gic_cols + mhs_cols

weekly_cols_dict = {
     'phq9': phq9_cols,
     'sds': sds_cols, 
     'sleep': sleep_cols,
     'gic': gic_cols, 
     'mhs': mhs_cols
}


drop_weekly_cols = []
for cols in weekly_cols_dict:
        if cols == 'phq9':
                for col in weekly_cols_dict[cols]:
                        for i in [0,5,7,9,11]: #keep weeks 1,2,3,4,6,8,10,12
                                drop_weekly_cols.append(f"{col}_week{i}")
        if cols == 'sds':
                for col in weekly_cols_dict[cols]:
                        for i in [0, 5,6,7,8,9,10,11,12]: #keep weeks 1,2,3,4
                                drop_weekly_cols.append(f"{col}_week{i}")
        if cols in ['sleep','mhs']:
                for col in weekly_cols_dict[cols]:
                        for i in [1,2,3,5,6,7,9,10,11]: #keep weeks 0,4,8,12
                                drop_weekly_cols.append(f"{col}_week{i}")
        if cols == 'gic':
                for col in weekly_cols_dict[cols]:
                        for i in [0, 4,5,6,7,8,9,10,11,12]: #keep weeks 1,2,3
                                drop_weekly_cols.append(f"{col}_week{i}")


weekly_cols_tuples = []
for col in weekly_cols:
        for i in range(0,13,1):
                weekly_cols_tuples.append(f"{col}_week{i}")
weekly_cols_tuples = set(weekly_cols_tuples) - set(drop_weekly_cols)


#### Variables from Daily Surveys
phq2_cols = ['phq2_1','phq2_2','phq2_sum']

daily_cols_v1 = ['aggregate_communication', 'call_count',
       'call_duration', 'interaction_diversity', 'missed_interactions',
       'sms_count', 'sms_length','mobility', 'mobility_radius'
       'unreturned_calls']

daily_v2_sensor = ['distance_walking', 'hours_active', 'distance_active',
        'came_to_work','distance_powered_vehicle',
       'hours_high_speed_transportation', 'hours_of_sleep',
       'distance_high_speed_transportation',
       'hours_powered_vehicle', 'location_variance']

daily_v2_phone = ['callDuration_incoming','callCount_missed',
        'callCount_outgoing','callCount_incoming',
       'callDuration_outgoing', 'textCount', 'textCount_received',
       'textCount_sent', 'textLength_received', 'textLength_sent',
       'uniqueNumbers_calls_incoming', 'uniqueNumbers_calls_missed',
       'uniqueNumbers_calls_outgoing', 'uniqueNumbers_texts',
       'uniqueNumbers_texts_received', 'uniqueNumbers_texts_sent']

daily_v2_weather = ['cloud_cover_mean','dew_point_mean',
        'humidity_mean','temp_mean','dew_point_IQR','humidity_IQR',
        'temp_IQR','cloud_cover_IQR','cloud_cover_std','dew_point_std',
        'humidity_std','temp_std','cloud_cover_median','dew_point_median',
        'humidity_median','temp_median','precip_sum']

daily_v2_common = ['distance_walking', 'hours_active', 'distance_active',
        'distance_powered_vehicle','hours_of_sleep','hours_powered_vehicle',
          'hours_stationary', 'hours_stationary_nhw','hours_walking']

daily_cols_v2 = daily_v2_sensor + daily_v2_phone + daily_v2_weather


daily_misc_cols = 'hours_accounted_for'
mobility_cols = ['mobility','mobility_radius']

daily_v2_sensor_avg = [f'{var}_hr' for var in daily_v2_sensor if var!='came_to_work']

all_daily_cols = phq2_cols + daily_cols_v1 + daily_v2_sensor_avg + daily_v2_phone + daily_v2_weather 


weather_cols = daily_v2_weather

phone_cols = ['aggregate_communication', 'call_count',
       'call_duration', 'interaction_diversity', 'missed_interactions','sms_count', 'sms_length',
       'unreturned_calls']

mobility_cols = ['mobility', 'mobility_radius','distance_walking', 
                 'hours_active', 'distance_active','hours_stationary', 
                 'hours_stationary_nhw','hours_walking']

places_cols = ['came_to_work','distance_high_speed_transportation','hours_high_speed_transportation', 
       'hours_powered_vehicle', 'location_variance','distance_powered_vehicle',]

sleep_cols = ['hours_of_sleep']

# Aggregated variable lists
numeric_cols = daily_cols_v1 + daily_v2_sensor + daily_v2_phone + phq2_cols 
passive_cols = daily_cols_v1 + daily_v2_sensor + daily_v2_phone

# df_list_all = df_names + aggregate_dfs
# id_key = pd.read_csv(os.path.join(brighten_dir, 'id_key.csv'))

# for name in df_list_all:
#     df = pd.read_csv(os.path.join(brighten_dir, f'{name}.csv'))
#     df_key = pd.merge(df, id_key[['num_id','v']], how='outer', on='num_id')
#     df_key.to_csv(os.path.join(brighten_dir, f'{name}_{end}.csv'))
#     print(f'Saved {name}.csv with V1/V2')

#     for end in df_endings:
#         if os.path.exists(os.path.join(brighten_dir, f'{name}_{end}.csv')):
#             df = pd.read_csv(os.path.join(brighten_dir, f'{name}_{end}.csv'))
#             df_key = pd.merge(df, id_key[['num_id','v']], how='outer', on='num_id')
#             df_key.to_csv(os.path.join(brighten_dir, f'{name}_{end}.csv'))
#             print(f'Saved {name}_{end}.csv with V1/V2')
#         else:
#             print(f'Skipping {name}_{end}.csv, cant find')