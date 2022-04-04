SELECT COUNT(*) FROM census WHERE workclass = 'Private' AND marital_status = 'Married-spouse-absent' AND occupation = 'Adm-clerical' AND race = 'Other' AND sex = 'Female' AND capital_gain <= 4101 AND capital_loss <= 2467 AND hours_per_week <= 79 AND native_country = 'India';