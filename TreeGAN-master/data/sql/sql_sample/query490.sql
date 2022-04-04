SELECT COUNT(*) FROM census WHERE age <= 41 AND workclass = 'Private' AND education = 'Assoc-acdm' AND education_num <= 16 AND marital_status = 'Married-AF-spouse' AND occupation = 'Sales' AND relationship = 'Unmarried' AND race = 'Other' AND sex = 'Male' AND capital_gain <= 4101 AND capital_loss <= 2559 AND hours_per_week <= 97 AND native_country = 'Hungary';