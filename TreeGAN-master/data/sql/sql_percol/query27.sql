SELECT COUNT(*) FROM census WHERE workclass = 'Never-worked' AND education = 'Assoc-acdm' AND education_num <= 16.0 AND marital_status = 'Divorced' AND occupation = 'Priv-house-serv' AND sex = 'Male' AND hours_per_week <= 81.0 AND native_country = 'England';