SELECT COUNT(*) FROM census WHERE age <= 75 AND workclass = 'Never-worked' AND education = 'Assoc-acdm' AND education_num <= 13 AND marital_status = 'Married-AF-spouse' AND occupation = 'Prof-specialty' AND relationship = 'Not-in-family' AND race = 'Amer-Indian-Eskimo' AND sex = 'Male' AND hours_per_week <= 13 AND native_country = 'Taiwan';