SELECT COUNT(*) FROM census WHERE workclass = 'Self-emp-not-inc' AND education = '1st-4th' AND education_num <= 11 AND marital_status = 'Divorced' AND occupation = '?' AND relationship = 'Husband' AND sex = 'Male' AND hours_per_week <= 32 AND native_country = 'Iran';