SELECT COUNT(*) FROM census WHERE workclass = 'Self-emp-not-inc' AND education = '1st-4th' AND education_num <= 11 AND marital_status = 'Never-married' AND occupation = 'Prof-specialty' AND relationship = 'Wife' AND race = 'Other' AND sex = 'Female' AND capital_gain <= 7298 AND capital_loss <= 2179 AND native_country = 'Trinadad&Tobago';