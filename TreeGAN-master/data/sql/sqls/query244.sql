SELECT COUNT(*) FROM census WHERE workclass = 'Private' AND education = '11th' AND education_num <= 11 AND occupation = 'Craft-repair' AND relationship = 'Not-in-family' AND race = 'White' AND sex = 'Female' AND hours_per_week <= 92 AND native_country = 'Portugal';