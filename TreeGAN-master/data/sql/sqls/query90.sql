SELECT COUNT(*) FROM census WHERE workclass = 'Self-emp-inc' AND education = '1st-4th' AND occupation = 'Prof-specialty' AND relationship = 'Not-in-family' AND capital_loss <= 1977 AND hours_per_week <= 16 AND native_country = 'Ireland';