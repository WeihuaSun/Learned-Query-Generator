SELECT COUNT(*) FROM census WHERE workclass = 'Private' AND education = '11th' AND education_num <= 11 AND marital_status = 'Widowed' AND occupation = 'Protective-serv' AND relationship = 'Other-relative' AND race = 'Black' AND sex = 'Male' AND hours_per_week <= 32;