SELECT COUNT(*) FROM census WHERE age <= 75 AND workclass = 'Without-pay' AND education = 'Assoc-acdm' AND education_num <= 8 AND marital_status = 'Widowed' AND occupation = 'Sales' AND relationship = 'Unmarried' AND race = 'White' AND sex = 'Female' AND capital_gain <= 4101 AND capital_loss <= 2559 AND hours_per_week <= 97 AND native_country = 'Poland';