SELECT COUNT(*) FROM census WHERE workclass = 'State-gov' AND education = '9th' AND education_num <= 15 AND marital_status = 'Divorced' AND relationship = 'Not-in-family' AND race = 'Black' AND sex = 'Male' AND capital_gain <= 1639 AND capital_loss <= 1944 AND native_country = 'Japan';