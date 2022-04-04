SELECT COUNT(*) FROM census WHERE workclass = 'Local-gov' AND education = 'Assoc-acdm' AND education_num <= 11 AND marital_status = 'Married-spouse-absent' AND occupation = 'Adm-clerical' AND relationship = 'Own-child' AND race = 'Asian-Pac-Islander' AND sex = 'Female' AND capital_gain <= 14344 AND capital_loss <= 2467 AND hours_per_week <= 79;