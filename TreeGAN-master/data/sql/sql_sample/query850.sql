SELECT COUNT(*) FROM census WHERE workclass = 'Local-gov' AND education = 'Prof-school' AND marital_status = 'Married-AF-spouse' AND relationship = 'Unmarried' AND race = 'White' AND sex = 'Male' AND capital_gain <= 8614 AND capital_loss <= 1974 AND hours_per_week <= 85 AND native_country = 'England';