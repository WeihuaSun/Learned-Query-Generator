SELECT COUNT(*) FROM census WHERE workclass = 'Never-worked' AND education = 'Doctorate' AND education_num <= 16 AND marital_status = 'Married-spouse-absent' AND occupation = 'Prof-specialty' AND relationship = 'Not-in-family' AND race = 'Other' AND sex = 'Male' AND capital_gain <= 14344 AND capital_loss <= 2080 AND hours_per_week <= 23 AND native_country = 'El-Salvador';