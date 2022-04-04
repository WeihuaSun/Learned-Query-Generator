SELECT COUNT(*) FROM census WHERE age <= 75 AND workclass = 'Without-pay' AND education = 'Assoc-voc' AND education_num <= 13 AND occupation = 'Adm-clerical' AND relationship = 'Own-child' AND race = 'White' AND sex = 'Female' AND capital_gain <= 8614 AND capital_loss <= 2080 AND hours_per_week <= 97 AND native_country = 'Poland';