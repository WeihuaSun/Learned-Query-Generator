SELECT COUNT(*) FROM census WHERE workclass = 'Self-emp-not-inc' AND marital_status = 'Never-married' AND occupation = 'Handlers-cleaners' AND relationship = 'Unmarried' AND race = 'White' AND sex = 'Male' AND capital_gain <= 15831 AND capital_loss <= 3004 AND hours_per_week <= 77 AND native_country = 'Yugoslavia';
