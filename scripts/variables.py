## Lists of variable names from data dfs in Brighten Dataset V1 & V2
# List of df names
df_names = ['v1_day', 'v2_day', 'v1_week', 'v2_week']
df_mis = ['v1_day_mis','v2_day_mis','v1_week_mis','v2_week_mis']
df_names_with_mis = df_names + df_mis

# Variables to keep consistent over all dfs and to not alter in preprocessing
id_columns=['num_id','dt','week','day','week_id','idx','v','season','day_of_week','cohort']
demographic_columns = ['gender','age','num_id']

## Variables from Baseline Surveys
gad_cols = ['gad7_1','gad7_2','gad7_3','gad7_4','gad7_5','gad7_6','gad7_7',
            'gad7_8','gad7_sum','gad_cat']
phq9_base = ['phq9_1_base','phq9_2_base','phq9_3_base','phq9_4_base','phq9_5_base',
             'phq9_6_base','phq9_7_base','phq9_8_base','phq9_9_base','phq9_sum_base']
clinical = ['cid','bin_clin','bipolar','scz','screen_1','screen_2','screen_3','screen_4']
demographics = ['gender','education','working','income_satisfaction',
                'incomelastyear','marital_status','race','age',
                'heard_about_us','device','startdate','study_arm', 'study']
alc_cols = ['alc_1','alc_2','alc_3','alc_sum', 'alc_cat']
mhs_cols = ['mhs_1','mhs_2','mhs_3','mhs_4','mhs_5']

baseline_cols = gad_cols + phq9_base + clinical + demographics + alc_cols + mhs_cols


## Variables from Weekly Surveys
phq9_cols = ['phq9_1','phq9_2','phq9_3','phq9_4','phq9_5','phq9_6',
             'phq9_7','phq9_8','phq9_9','phq9_sum', 'phq9_bin','phq9_cat']
sds_cols = ['sds_1','sds_2','sds_3','stress','support']
sleep_cols = ['sleep_1','sleep_2','sleep_3']
gic_cols = ['mood_1']

weekly_cols = phq9_cols + sds_cols + sleep_cols + gic_cols + mhs_cols

weekly_cols_dict = {
     'phq9': phq9_cols, #keep weeks 1,2,3,4,6,8,10,12
     'sds': sds_cols, #keep weeks 1,2,3,4
     'sleep': sleep_cols, #keep weeks 0,4,8,12
     'gic': gic_cols, # keep all weeks
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
        if cols in ['sleep']:
                for col in weekly_cols_dict[cols]:
                        for i in [1,2,3,5,6,7,9,10,11]: #keep weeks 0,4,8,12
                                drop_weekly_cols.append(f"{col}_week{i}")


weekly_cols_tuples = []
for col in weekly_cols:
        for i in range(0,13,1):
                weekly_cols_tuples.append(f"{col}_week{i}")
weekly_cols_tuples = set(weekly_cols_tuples) - set(drop_weekly_cols)


#### Variables from Daily Surveys
phq2_cols = ['phq2_1','phq2_2','phq2_sum']

ordinal_cols = gad_cols + phq9_base + alc_cols + phq9_cols + sleep_cols + gic_cols + phq2_cols

daily_cols_v1 = ['aggregate_communication', 'call_count',
       'call_duration', 'interaction_diversity', 'missed_interactions',
       'sms_count', 'sms_length','mobility', 'mobility_radius',
       'unreturned_calls']

daily_v2_sensor = ['distance_walking', 'hours_active', 'distance_active',
        'came_to_work','distance_powered_vehicle',
       'hours_high_speed_transportation', 'hours_of_sleep',
       'distance_high_speed_transportation',
       'hours_powered_vehicle', 'location_variance']

daily_v2_sensor_hr = [f'{col}_hr' for col in daily_v2_sensor if col != 'came_to_work']

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
daily_cols_v2_hr = daily_v2_sensor + daily_v2_phone + daily_v2_weather

daily_misc_cols = 'hours_accounted_for'
mobility_cols = ['mobility','mobility_radius']

# Aggregated variable lists
numeric_cols = daily_cols_v1 + daily_v2_sensor_hr + daily_v2_phone + phq2_cols 
passive_cols = daily_cols_v1 + daily_v2_sensor_hr + daily_v2_phone


all_daily_cols = phq2_cols + daily_cols_v1 + daily_cols_v2_hr


## ALL
all_cols = id_columns + baseline_cols + weekly_cols + all_daily_cols



# By content
weather_cols_v2 = daily_v2_weather

phone_cols_v2 = ['aggregate_communication', 'call_count',
       'call_duration', 'interaction_diversity', 'missed_interactions','sms_count', 'sms_length',
       'unreturned_calls']

mobility_cols_v2 = ['mobility', 'mobility_radius','distance_walking', 
                 'hours_active', 'distance_active','hours_stationary', 
                 'hours_stationary_nhw','hours_walking']

places_cols_v2 = ['came_to_work','distance_high_speed_transportation','hours_high_speed_transportation', 
       'hours_powered_vehicle', 'location_variance','distance_powered_vehicle','hours_of_sleep']



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



'''
## Questions:

# Daily
### PHQ2
4 	dt_yesterday 	Date 	The date for which the mood of this survey represents
5 	day 	int 	The number of days into the study the participant was when they completed this survey
6 	phq2_1 	int 	Yesterday, were you bothered by any of the following problems: feeling down, depressed, or hopeless
7 	phq2_2 	int 	Yesterday, did you have little interest or pleasure in doing things?

        Answers:
        1 Not at all
        ... 
        5 Most of the day


# Weekly
### Impression of Change: Weekly
- mood_1    'Since beginning this study, how would you describe the change (if any) in ACTIVITY LIMITATIONS, SYMPTOMS, EMOTIONS, and OVERALL QUALITY OF LIFE, related to your mood? '
	* *1 = No change, 2 = Almost the same, 3 = A little better, 4 = Somewhat better, 5 = Much better, 6 = Very much better*

### PHQ9: Weeks 1,2,3,4,6,8,10,12
2 	phq9Date 	Date
3 	week 	The nthnth week into the study
4 	phq9_1 	Little interest or pleasure in doing things
5 	phq9_2 	Feeling down, depressed, or hopeless
6 	phq9_3 	Trouble falling or staying asleep, or sleeping too much
7 	phq9_4 	Feeling tired or having little energy
8 	phq9_5 	Poor appetite or overeating
9 	phq9_6 	Feeling bad about yourself or that you are a failure or have let yourself or your family down
10 	phq9_7 	Trouble concentrating on things, such as reading the newspaper or watching television
11 	phq9_8 	Moving or speaking so slowly that other people could have noticed? Or the opposite; being so fidgety or restless that you have been moving around a lot more than usual
12 	phq9_9 	Thoughts that you would be better off dead or of hurting yourself in some way

        Answers: 
        1 Not at all 
        2 Several days
        3 Over half the days
        4 Nearly every day



# Some weeks

### SDS: Weeks 1,2,3,4
- sds_1 	The symptoms have disrupted your work/school
- sds_2 	The symptoms have disrupted your social life / leisure activities
- sds_3 	The symptoms have disrupted your family life / home responsibilities
- stress 	In the last week, how much were you set back by stressful events or personal problems such as work, home, social, health, or financial problems?
- support In the past week, how much support have you received from friends, relatives, co-workers, etc., as a percentage of the amount you needed to cope?
	* *0 = not at all, 3 = Mildly, 6 = Moderately, 9 = Severely*

### Sleep: Weeks 0,4,8,12
- sleep_1 	Thinking about your time in bed at night, over the past week, on average how long did it take you to fall asleep?
- sleep_3 	Thinking about your time in bed at night, over the past week, on average how long did you lie awake after having been asleep?
	* *1 = 0 - 15 minutes, 2 = 16 - 30 minutes, 3 = 31 - 60 minutes, 4 = 61 - 90 minutes, 5 = 91 - 120 minutes, 6 = More than 120 minutes*
- sleep_2 	Thinking about your time in bed at night, over the past week, on average how long were you actually asleep?
	* *1 = 5-6 hours, 2 = 6-7 hours, 3 = 7-8 hours, 4 = Less than 5 hours, 5 = More than 8 hours*


# Baseline
### Demographics
2 	gender 	str 	an element ∈∈ { Male, Female }  
3 	education 	str 	an element ∈∈ { University, High School, Community College, Graduate Degree, Elementary School, None }  
4 	working 	str 	an element ∈∈ { Yes, No }  
5 	income_satisfaction 	str 	an element ∈∈ { Can't make ends meet, Have enough to get along, Am comfortable }  
6 	incomelastyear 	str 	elements are discretized into 6 bins separated by 20K upto 100K with the 6th being anything greater
7 	marital_status 	str 	an element ∈∈ { Single, Married/Partner, Separated/Widowed/Divorced }  
8 	race 	str 	an element ∈∈ { African-American/Black, American Indian/Alaskan Native, Asian, Hispanic/Latino, More than one, Native Hawaiian/other Pacific Islander, Non-Hispanic White, Other }  
9 	age 	int 	The age of the participant  
10 	heardaboutus 	str 	an element ∈∈ { Advertisement, Craigslist, Twitter/Facebook, friend/colleague, others, through other studies }  
11 	device 	str 	an element ∈∈ { IPhone, Android }  
12 	start_date 	DateTime 	The UTC date and time for when the participant began the study  
13 	study_arm 	str 	an element ∈∈ { EVO, HealthTips, iPST , NA - did not get randomized}  
14 	study 	str 	an element ∈∈ { Brighten-v2, Brighten-v1 }  

### Mental Health Services
- mhs_1 	An answer to, 'Do you see a psychiatrist for mental health or substance abuse treatment?'
- mhs_2 	An answer to, 'Do you see a psychologist, social worker, or other counselors for mental health or substance abuse treatment? '
- mhs_3 	An answer to, 'Do you attend any groups for mental health or substance abuse treatment?'
- mhs_4 	An answer to, ' Are you reading any self-help books for mental health or substance abuse treatment?'
- mhs_5 	An answer to, 'Are you taking any medications for mental health problems? 
	* *-1 = Decline, 0 = No, 1 = Yes* 

### Other Health-related Apps Used
This questionnaire was sent at timepoints week 0, 4, 8, and 12 and posed the question, “We want to learn more about the kinds of apps people download for health, mood, and mental health improvement reasons. Please indicate which apps you have downloaded and use regularly for a. Mood; b. Concentration; c. Relaxation; d. Sleep; e. Pain management; f. Weight management; g. Alcohol; h. Exercise/fitness; and/or i. Medical.

4 	app_al 	int 	an indicator for Alcohol  
5 	app_ct 	int 	an indicator for Concentration  
6 	app_ef 	int 	an indicator for Exercise and Fitness  
7 	app_md 	int 	an indicator for Medical  
8 	app_mo 	int 	an indicator for Mood  
9 	app_pm 	int 	an indicator for Pain Management  
10 	app_rx 	int 	an indicator for Relaxation  
11 	app_sl 	int 	an indicator for Sleep  
12 	app_wm 	int 	an indicator for Weight Management  


        
### GAD 7
4 	gad7_1 	int 	Feeling nervous, anxious, or on edge
5 	gad7_2 	int 	Not being able to stop or control worrying
6 	gad7_3 	int 	Worrying too much about different things
7 	gad7_4 	int 	Trouble relaxing
8 	gad7_5 	int 	Being so restless that it's hard to sit still
9 	gad7_6 	int 	Becoming easily annoyed or irritable
10 	gad7_7 	int 	Feeling afraid as if something awful might happen
11 	gad7_8 	int 	 "If you checked off any problems, how difficult have these made it for you to do your work, take care of things at home, or get along with other people?"
12 	gad7_sum 	int 	sum of 1-8

        Answer (gad 1 - 7)
        1 Not at all sure
        2 Several days
        3 Over half the days
        4 Nearly every day

        Answer (gad 8)
        1 Not difficult at all
        2 Somewhat difficult
        3 Very difficult
        4 Extremely difficult




### Download app reason
This one question, multiple choice survey asked “Why did you download this app?”. The potential answers were a) for fun; b) for mental health reasons [e.g. depression; anxiety; ADHD]; c) for mood [e.g. sadness]; d) for management of daily problems; e) for brain health [e.g. better memory]; f) to improve relationships; g) to improve work; h) other [type in box], with the ability to select multiple answers if desired. This survey was collect at Week 0.

happ_bh 	int 	an indicator for brain health  
happ_f 	int 	an indicator for fun  
happ_fts 	int 	an indicator for did it for the study.   
happ_inc 	int 	an indicator for financial incentives.     
happ_ir 	int 	an indicator for improve relationships.   
happ_iw 	int 	an indicator for improve work.   
happ_m 	int 	an indicator for mood.   
happ_mdi 	int 	an indicator for manage daily issues.   
happ_mh 	int 	an indicator for mental health.   
happ_o 	int 	an indicator for other.   
happodescription 	str 	Free text input if other was selected.   


### Mania and psychosis screening
screen_1 	int 	an answer to, 'Did a doctor ever prescribe a medication called Lithium to you?'  
screen_2 	int 	an answer to, 'Were you prescribed any medication for having a period of being so excited or irritable that you got into trouble or your family or friends worried about it?'  
screen_3 	int 	an answer to, 'Did a doctor ever say you were manic-depressive or had bipolar disorder?'  
screen_4 	int 	an answer to, 'Did a doctor ever say that you have schizophrenia or a schizoaffective disorder or psychosis?'  


### Alcohol AUDIT-C
2 	dt_response 	DateTime 	The DateTime representing when this survey was completed
3 	week 	int 	The nthnth week into the study
4 	alc_1 	int 	An answer to 'How often did you have a drink containing alcohol in the past year?'
5 	alc_2 	int 	An answer to 'How many drinks did you have on a typical day when you were drinking in the past year?'
6 	alc_3 	int 	An answer to 'How often did you have six or more drinks on one occasion in the past year?' 

        Answers: 
        alc_1: 
        0 Never
        1 Monthly or less 
        2 2-4 times a mont
        3 2-3 times a week
        4 4 or more times a week

        alc_2: 
        0 1 or 2
        1 3 to 4
        2 5 to 6
        3 7 to 9
        4 10 or more

        alc_3
        0 Never
        1 Less than monthly
        2 Monthly
        3 Weekly
        4 Daily or almost daily



        
'''


'''
Passive Data

V1 
participant_id 	str 	Unique ID
dt_passive 	DateTime 	The UTC date and time representing when this survey was completed
day 	int 	Days number of study
week 	int 	Week number of study
cohort 	str 	an element ∈∈ {PST, Akili, Health tips}
userphonetype 	str 	an element ∈∈ {iPhone, Android}

aggregate_communication 	int 	
        Total number of calls and total number of SMS messages on a particular day

call_count 	int 	Total number of calls

call_duration 	int 	Total duration of all calls in seconds

interaction_diversity 	int 	
                Total number of unique individuals with whom a 
                participant interacted through phone calls or SMS messages on a particular day

missed_interactions 	int 	Total number of calls unanswered for a user on a particular day

mobility 	float 	
        Approximate distance in miles covered by the user by foot or by 
        bike on a particular day as determined from location data

mobility_radius 	float 	
        Approximate radius of an imaginary circle encompassing the various
        locations that a user has traveled across on a particular day, in miles

sms_count 	int 	Number of SMS messages sent and received

sms_length 	int 	Total number of characters from both sent and received

unreturned_calls 	int 	The number of missed calls without an associated call back



'''


'''
V2
1 	participant_id 	str 	Unique ID
2 	dt_passive 	Date 	Date of aggregated data
3 	week 	int 	The nthnth week into the study for this response

4 	cametowork 	bool 	∈∈ { true, false } 
                indicating whether or not the participant came to work

5 	distance_active 	float 	
                The cumulative distance traveled in the active velocity bin (meters)

6 	distancehighspeed_transportation 	float 	
                The cumulative distance traveled in the high speed transportation velocity bin (meters)

7 	distancepoweredvehicle 	float 	
                The cumulative distance traveled in the powered vehicle velocity bin (meters)

8 	distance_walking 	float 	
                The cumulative distance traveled in the walking velocity bin (meters)

9 	hoursaccountedfor 	int 	
                The count of hours in the day for which GPS records exist

10 	hours_active 	float 	
                The cumulative time spent in the active velocity bin (hours)

11 	hourshighspeed_transportation 	float 	
                The cumultive time spent in the high speed transportation velocity bin (hours)

12 	hoursofsleep 	float 	
                The estimated hours of sleep accrued during the previous night (hours)

13 	hourspoweredvehicle 	float 	
                The cumulative time spent in the powered vehicle velocity bin (hours)

14 	hours_stationary 	float 	
                The cumulative time spent in the stationary velocity bin (hours)

15 	hoursstationarynhw 	float 	
                The cumulative time spent in the stationary velocity bin 
                excluding the time spent in the home and work clusters (hours)

16 	hours_walking 	float 	
                The cumulative time spent in the walking velocity bin (hours)

17 	location_variance 	float 	log(variance^2 of longitude + variance^2 of latitude)

'''