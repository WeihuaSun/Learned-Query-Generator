SELECT COUNT(*) FROM census WHERE workclass = 'Self-emp-inc' AND education_num <= 7 AND marital_status = 'Married-spouse-absent' AND relationship = 'Unmarried' AND race = 'White' AND sex = 'Female' AND capital_gain <= 20051 AND capital_loss <= 2174 AND hours_per_week <= 92;